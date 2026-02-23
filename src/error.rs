use std::{
    ffi::NulError,
    fmt::{Arguments, Display},
    io::{Error, Write as _, stderr},
    num::{ParseFloatError, ParseIntError},
};

use libc::_exit;
use thiserror::Error;

use crate::error::Xerr::Msg;

#[derive(Error, Debug)]
pub enum Xerr {
    #[error("{0}")]
    Io(#[from] Error),

    #[error("{0}")]
    Ffi(#[from] NulError),

    #[error("{0}")]
    ParseInt(#[from] ParseIntError),

    #[error("{0}")]
    ParseFloat(#[from] ParseFloatError),

    #[error("{0}")]
    Msg(String),

    #[error("")]
    Help,

    #[error("")]
    Done,
}

impl From<&str> for Xerr {
    fn from(s: &str) -> Self {
        Msg(s.into())
    }
}

impl From<String> for Xerr {
    fn from(s: String) -> Self {
        Msg(s)
    }
}

#[cold]
#[inline(never)]
pub fn fatal<E: Display>(e: E) -> ! {
    _ = writeln!(stderr(), "{e}");
    unsafe { _exit(1) }
}

#[cold]
#[inline(never)]
pub fn eprint(args: Arguments<'_>) {
    _ = writeln!(stderr(), "{args}");
}
