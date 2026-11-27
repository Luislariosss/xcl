//! Physical pages mapped into the title's own reservations, committed on first use.
//!
//! A title maps its physical pages wherever it reserved address space and places resources
//! there – render targets and depth buffers too, which on the PC live in video memory
//! (`d3d12x::vram`); the CPU never touches those pages. Committed at map time they doubled
//! that memory (a retail game: 5.4 GB of mapped pages next to its video-memory targets, past
//! the system's commit). So a page whose contents are still zero is mapped as reserved address
//! space only: the CPU's first access faults, the page is committed (zeros, as the page holds)
//! and the access goes on. Kernel calls do not fault in the title – they fail – so the buffers
//! of the file calls XCL routes, and DirectStorage's destinations, are committed first
//! (`ensure`).

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Once, RwLock};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
};

const PAGE: usize = 0x1_0000;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// pages (64 KiB) mapped but not committed yet
static LAZY: RwLock<BTreeSet<usize>> = RwLock::new(BTreeSet::new());
/// Fast-path count of `LAZY` (avoids taking the lock when no lazy pages exist).
static LAZY_COUNT: AtomicUsize = AtomicUsize::new(0);

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

/// Commit `page` if it is a lazy one (true: it was).
unsafe fn commit(page: usize) -> bool {
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    if !g.remove(&page) {
        return false;
    }
    let ok = !VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null();
    LAZY_COUNT.store(g.len(), Ordering::Release);
    ok
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let page = r.ExceptionInformation[1] & !(PAGE - 1);
    let lazy = LAZY.read().unwrap_or_else(|e| e.into_inner()).contains(&page);
    if lazy && commit(page) || allows(r.ExceptionInformation[0], r.ExceptionInformation[1]) {
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

unsafe fn allows(access: usize, at: usize) -> bool {
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT || m.Protect & PAGE_GUARD != 0 {
        return false;
    }
    let writes = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    let reads = writes | PAGE_READONLY | PAGE_EXECUTE_READ;
    match access {
        0 => m.Protect & reads != 0,
        1 => m.Protect & writes != 0,
        _ => false,
    }
}

/// `page` (reserved, 64 KiB aligned) is mapped: committed on first use.
pub(crate) fn mark(page: usize) {
    install();
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    if g.insert(page) {
        LAZY_COUNT.store(g.len(), Ordering::Release);
    }
}

/// Mark every 64 KiB page in `[start, start + len)` for demand-commit on first touch.
pub(crate) fn mark_range(start: usize, len: usize) {
    if len == 0 {
        return;
    }
    install();
    let first = start & !(PAGE - 1);
    let end = start.saturating_add(len).div_ceil(PAGE).saturating_mul(PAGE);
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    let mut p = first;
    while p < end {
        g.insert(p);
        match p.checked_add(PAGE) {
            Some(next) => p = next,
            None => break,
        }
    }
    LAZY_COUNT.store(g.len(), Ordering::Release);
}

/// `page` is unmapped: true if it was never used (nothing to keep of it).
pub(super) fn unmark(page: usize) -> bool {
    if LAZY_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    let removed = g.remove(&page);
    if removed {
        LAZY_COUNT.store(g.len(), Ordering::Release);
    }
    removed
}

/// Unmark every 64 KiB lazy page overlapping `[start, start + len)`.
pub(crate) fn unmark_range(start: usize, len: usize) {
    if len == 0 || LAZY_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let first = start & !(PAGE - 1);
    let end = start.saturating_add(len);
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    let to_remove: Vec<usize> = g.range(first..end).copied().collect();
    for p in to_remove {
        g.remove(&p);
    }
    LAZY_COUNT.store(g.len(), Ordering::Release);
}

/// True if `addr` falls inside a demand-commit lazy page.
pub(crate) fn is_marked(addr: usize) -> bool {
    if LAZY_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let page = addr & !(PAGE - 1);
    LAZY.read().unwrap_or_else(|e| e.into_inner()).contains(&page)
}

/// The lazy pages of `[p, p + len)` committed – memory a kernel call or XCL writes.
/// Also brings back any CPU-parked sparse-window tiles in `[p, p + len)` from VRAM.
pub fn ensure(p: usize, len: usize) {
    if len == 0 {
        return;
    }
    unsafe { crate::hle::d3d12x::sparse::ensure_cpu_resident(p, len) };
    if LAZY_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let pages: Vec<usize> = {
        let g = LAZY.read().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            return;
        }
        g.range(p & !(PAGE - 1)..p.saturating_add(len)).copied().collect()
    };
    for page in pages {
        unsafe { commit(page) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE, MEM_RESERVE};

    /// A lazy page reads as zeros on first touch, keeps what is written; `ensure` commits it
    /// without a fault.
    #[test]
    fn a_lazy_page_is_committed_on_first_use() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark(r);
            mark(r + PAGE);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 0);
            std::ptr::write_volatile((r + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 7);
            assert!(!unmark(r), "used: no longer lazy");
            ensure(r + PAGE + 4, 4);
            assert!(!LAZY.read().unwrap().contains(&(r + PAGE)));
            assert!(crate::gpu::shader::readable(r + PAGE, 4));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn threads_first_touching_one_lazy_page_together_all_go_on() {
        const THREADS: usize = 8;
        const PAGES: usize = 256;
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            for i in 0..PAGES {
                mark(r + i * PAGE);
            }
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        for i in 0..PAGES {
                            barrier.wait();
                            std::ptr::write_volatile((r + i * PAGE + t * 64) as *mut u64, (i * THREADS + t) as u64);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            for i in 0..PAGES {
                for t in 0..THREADS {
                    assert_eq!(std::ptr::read_volatile((r + i * PAGE + t * 64) as *const u64), (i * THREADS + t) as u64);
                }
            }
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn mark_range_and_unmark_range_track_64k_pages() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark_range(r, 4 * PAGE);
            assert!(is_marked(r));
            assert!(is_marked(r + 3 * PAGE + 123));
            unmark_range(r + PAGE, 2 * PAGE);
            assert!(is_marked(r));
            assert!(!is_marked(r + PAGE));
            assert!(!is_marked(r + 2 * PAGE));
            assert!(is_marked(r + 3 * PAGE));
            unmark_range(r, 4 * PAGE);
            assert!(!is_marked(r));
            assert!(!is_marked(r + 3 * PAGE));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
