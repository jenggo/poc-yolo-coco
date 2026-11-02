use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// A bounded queue that drops its oldest entry when a producer overruns a slow
/// consumer.
///
/// `std::sync::mpsc` is unbounded, so a slow inference stage would grow latency
/// without limit and the window would show increasingly stale video. Dropping the
/// oldest frame keeps display latency constant and discards frames nobody will
/// look at.
pub struct Mailbox<T> {
    shared: Mutex<Shared<T>>,
    available: Condvar,
    capacity: usize,
}

struct Shared<T> {
    items: VecDeque<T>,
    closed: bool,
    dropped: u64,
}

impl<T> Mailbox<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            shared: Mutex::new(Shared {
                items: VecDeque::with_capacity(capacity),
                closed: false,
                dropped: 0,
            }),
            available: Condvar::new(),
            capacity: capacity.max(1),
        }
    }

    /// A poisoned lock only means another thread panicked; the queue itself is
    /// still consistent, so recover instead of propagating the panic.
    fn lock(&self) -> MutexGuard<'_, Shared<T>> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Push an item, dropping the oldest one if the queue is full.
    ///
    /// Returns `false` when the mailbox is closed and the item was discarded.
    pub fn push_drop_oldest(&self, item: T) -> bool {
        let mut shared = self.lock();
        if shared.closed {
            return false;
        }
        if shared.items.len() >= self.capacity {
            shared.items.pop_front();
            shared.dropped += 1;
        }
        shared.items.push_back(item);
        drop(shared);
        self.available.notify_one();
        true
    }

    /// Block until an item is available, or return `None` once the mailbox is
    /// closed and drained.
    pub fn pop_blocking(&self) -> Option<T> {
        let mut shared = self.lock();
        loop {
            if let Some(item) = shared.items.pop_front() {
                return Some(item);
            }
            if shared.closed {
                return None;
            }
            shared = self
                .available
                .wait(shared)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Return an item without blocking, or `None` when the queue is empty.
    pub fn try_pop(&self) -> Option<T> {
        self.lock().items.pop_front()
    }

    /// Close the mailbox and wake every waiting consumer.
    pub fn close(&self) {
        let mut shared = self.lock();
        shared.closed = true;
        drop(shared);
        self.available.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }

    pub fn len(&self) -> usize {
        self.lock().items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total number of items discarded because the queue was full.
    pub fn dropped(&self) -> u64 {
        self.lock().dropped
    }
}

/// Cooperative shutdown flag shared by every pipeline stage.
#[derive(Clone, Default)]
pub struct Shutdown(Arc<AtomicBool>);

impl Shutdown {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn trigger(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_triggered(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn drop_oldest_keeps_the_newest_items() {
        let mailbox = Mailbox::new(2);
        mailbox.push_drop_oldest(1);
        mailbox.push_drop_oldest(2);
        mailbox.push_drop_oldest(3);

        assert_eq!(mailbox.len(), 2);
        assert_eq!(mailbox.dropped(), 1);
        assert_eq!(mailbox.try_pop(), Some(2));
        assert_eq!(mailbox.try_pop(), Some(3));
        assert_eq!(mailbox.try_pop(), None);
    }

    #[test]
    fn pop_blocking_waits_for_a_push() {
        let mailbox = Arc::new(Mailbox::new(1));
        let producer = {
            let mailbox = Arc::clone(&mailbox);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                mailbox.push_drop_oldest(7);
            })
        };

        assert_eq!(mailbox.pop_blocking(), Some(7));
        producer.join().unwrap();
    }

    #[test]
    fn close_wakes_a_blocked_consumer() {
        let mailbox: Arc<Mailbox<u32>> = Arc::new(Mailbox::new(1));
        let consumer = {
            let mailbox = Arc::clone(&mailbox);
            thread::spawn(move || mailbox.pop_blocking())
        };

        thread::sleep(Duration::from_millis(20));
        mailbox.close();
        assert_eq!(consumer.join().unwrap(), None);
    }

    #[test]
    fn close_drains_before_returning_none() {
        let mailbox = Mailbox::new(4);
        mailbox.push_drop_oldest(1);
        mailbox.close();
        assert_eq!(mailbox.try_pop(), Some(1));
        assert_eq!(mailbox.pop_blocking(), None);
    }

    #[test]
    fn pushing_after_close_is_refused() {
        let mailbox = Mailbox::new(1);
        mailbox.close();
        assert!(!mailbox.push_drop_oldest(1));
        assert!(mailbox.is_empty());
    }

    #[test]
    fn shutdown_flag_is_shared() {
        let flag = Shutdown::new();
        let clone = flag.clone();
        assert!(!flag.is_triggered());
        clone.trigger();
        assert!(flag.is_triggered());
    }
}
