//! A minimal C-ABI wrapper so a browser (or any WebAssembly host) can run the
//! swadesh CLI on in-memory files. No wasm-bindgen needed.
//!
//! Protocol: allocate with `sw_alloc`, copy bytes in, register files with
//! `sw_set_file(name, data)` (name `"<stdin>"` is standard input), call
//! `sw_run(argv)` with arguments separated by NUL bytes, then read stdout via
//! `sw_out_ptr`/`sw_out_len` and the error message via `sw_err_ptr`/`sw_err_len`.

use std::cell::RefCell;

thread_local! {
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static ERR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// # Safety
/// Returns a buffer of `len` bytes owned by the caller (free with `sw_free`).
#[no_mangle]
pub extern "C" fn sw_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `p`/`len` must come from `sw_alloc`.
#[no_mangle]
pub unsafe extern "C" fn sw_free(p: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(p, 0, len.max(1)));
}

unsafe fn bytes<'a>(p: *const u8, len: usize) -> &'a [u8] {
    if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, len)
    }
}

/// # Safety
/// Pointers must reference valid memory of the given lengths.
#[no_mangle]
pub unsafe extern "C" fn sw_set_file(name: *const u8, name_len: usize, data: *const u8, data_len: usize) {
    let name = String::from_utf8_lossy(bytes(name, name_len)).into_owned();
    swadesh_cli::input::set_virtual_file(&name, bytes(data, data_len).to_vec());
}

#[no_mangle]
pub extern "C" fn sw_clear_files() {
    swadesh_cli::input::clear_virtual_files();
}

/// Run the CLI. Returns the process exit code (0 = success).
///
/// # Safety
/// `argv` must reference `argv_len` valid bytes (NUL-separated UTF-8 arguments).
#[no_mangle]
pub unsafe extern "C" fn sw_run(argv: *const u8, argv_len: usize) -> i32 {
    let raw = String::from_utf8_lossy(bytes(argv, argv_len)).into_owned();
    let args: Vec<String> = if raw.is_empty() { Vec::new() } else { raw.split('\0').map(str::to_string).collect() };
    let mut out = Vec::new();
    let res = swadesh_cli::run_argv(&args, &mut out);
    OUT.with(|o| *o.borrow_mut() = out);
    match res {
        Ok(()) => {
            ERR.with(|e| e.borrow_mut().clear());
            0
        }
        Err(f) => {
            ERR.with(|e| *e.borrow_mut() = format!("swadesh: error: {}", f.msg).into_bytes());
            f.code
        }
    }
}

#[no_mangle]
pub extern "C" fn sw_out_ptr() -> *const u8 {
    OUT.with(|o| o.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn sw_out_len() -> usize {
    OUT.with(|o| o.borrow().len())
}
#[no_mangle]
pub extern "C" fn sw_err_ptr() -> *const u8 {
    ERR.with(|e| e.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn sw_err_len() -> usize {
    ERR.with(|e| e.borrow().len())
}
