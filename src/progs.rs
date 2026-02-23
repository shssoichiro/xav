use std::{
    io::{BufRead as _, BufReader, Read},
    str::from_utf8,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread::{JoinHandle, sleep, spawn},
    time::{Duration, Instant},
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::{
    chunk::{Chunk, PRIOR_SECS},
    encoder::{
        Encoder,
        Encoder::{Avm, SvtAv1, Vvenc, X264, X265},
    },
    error::eprint,
    ffms::VidInf,
};

const BAR_WIDTH: usize = 20;
const INTERVAL_MS: u64 = 500;

use crate::util::{B, C, G, N, P, R, W, Y};

const G_HASH: &str = "\x1b[1;92m#";
const R_DASH: &str = "\x1b[1;91m-";
const B_HASH: &str = "\x1b[1;94m#";
const Y_DASH: &str = "\x1b[1;93m-";

fn fmt_el(h: usize, m: usize) -> String {
    if h == 0 && m == 0 {
        String::new()
    } else {
        format!("{W}{h:02}{P}:{W}{m:02} ")
    }
}

fn fmt_eta(h: usize, m: usize) -> String {
    if h == 0 && m == 0 {
        String::new()
    } else {
        format!("{C}, {W}-{h:02}{P}:{W}{m:02}")
    }
}

fn msg_style() -> ProgressStyle {
    unsafe { ProgressStyle::with_template("{msg}").unwrap_unchecked() }
}

pub struct ProgsBar {
    scd_bar: ProgressBar,
    audio_bar: ProgressBar,
    start: Instant,
    last_update: Instant,
}

impl ProgsBar {
    pub fn new() -> Self {
        let scd_bar = ProgressBar::new_spinner();
        scd_bar.set_style(msg_style());
        let audio_bar = ProgressBar::new_spinner();
        audio_bar.set_style(msg_style());
        Self {
            scd_bar,
            audio_bar,
            start: Instant::now(),
            last_update: Instant::now(),
        }
    }

    pub fn up_scenes(&mut self, current: usize, total: usize, _line: usize) {
        if self.last_update.elapsed() < Duration::from_millis(INTERVAL_MS) {
            return;
        }
        self.last_update = Instant::now();

        let elapsed = self.start.elapsed().as_secs() as usize;
        let fps = current / elapsed.max(1);
        let remaining = total.saturating_sub(current);
        let eta_secs = remaining * elapsed / current.max(1);
        let filled = (BAR_WIDTH * current / total.max(1)).min(BAR_WIDTH);
        let bar = format!(
            "{}{}",
            G_HASH.repeat(filled),
            R_DASH.repeat(BAR_WIDTH - filled)
        );
        let perc = (current * 100 / total.max(1)).min(100);
        let el = fmt_el(elapsed / 3600, (elapsed % 3600) / 60);
        let eta = fmt_eta(eta_secs / 3600, (eta_secs % 3600) / 60);

        self.scd_bar.set_message(format!(
            "{el}{W}SCD: {C}[{bar}{C}] {W}{perc}%{C}, {Y}{fps} FPS{eta}{C}, \
             {G}{current}{C}/{R}{total}{N}"
        ));
    }

    pub fn up_scenes_final(&mut self, total: usize, line: usize) {
        self.last_update = unsafe {
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_unchecked()
        };
        self.up_scenes(total, total, line);
    }

    pub fn up_audio(
        &mut self,
        current: usize,
        total: usize,
        _line: usize,
        pass: u8,
        track_id: usize,
    ) {
        if self.last_update.elapsed() < Duration::from_millis(INTERVAL_MS) {
            return;
        }
        self.last_update = Instant::now();

        let elapsed = self.start.elapsed().as_secs() as usize;
        let speed = current as f64 / elapsed.max(1) as f64 / 48000.0;
        let remaining = total.saturating_sub(current);
        let eta_secs = remaining * elapsed / current.max(1);
        let filled = (BAR_WIDTH * current / total.max(1)).min(BAR_WIDTH);
        let bar = format!(
            "{}{}",
            G_HASH.repeat(filled),
            R_DASH.repeat(BAR_WIDTH - filled)
        );
        let perc = (current * 100 / total.max(1)).min(100);
        let el = fmt_el(elapsed / 3600, (elapsed % 3600) / 60);
        let eta = fmt_eta(eta_secs / 3600, (eta_secs % 3600) / 60);
        let dur = total / 48000;
        let (dh, dm, ds) = (dur / 3600, (dur % 3600) / 60, dur % 60);

        self.audio_bar.set_message(format!(
            "{W}{track_id:02}{C}] {el}{W}AU P{pass}: {C}[{bar}{C}] {W}{perc}%{C}, \
             {Y}{speed:.1}x{eta}{C}, {G}{dh:02}{P}:{G}{dm:02}{P}:{G}{ds:02}{N}"
        ));
    }

    pub fn up_audio_final(&mut self, total: usize, line: usize, pass: u8, track_id: usize) {
        self.last_update = unsafe {
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_unchecked()
        };
        self.up_audio(total, total, line, pass, track_id);
    }

    pub fn finish_audio(&self) {
        self.audio_bar.finish_and_clear();
    }

    pub fn finish_scenes(&self) {
        self.scd_bar.finish_and_clear();
    }
}

struct SummaryState {
    total_chunks: usize,
    total_frames: usize,
    fps_num: usize,
    fps_den: usize,
    completed: Arc<AtomicUsize>,
    completed_frames: Arc<AtomicUsize>,
    total_size: Arc<AtomicU64>,
    processed: AtomicUsize,
    start: Instant,
    init_frames: usize,
}

pub struct ProgsTrack {
    #[allow(dead_code)]
    multi: MultiProgress,
    worker_bars: Vec<ProgressBar>,
    summary_bar: ProgressBar,
    state: Arc<SummaryState>,
    stop: Arc<AtomicBool>,
    ticker: Option<JoinHandle<()>>,
}

impl ProgsTrack {
    pub fn new(
        chunks: &[Chunk],
        inf: &VidInf,
        worker_count: usize,
        init_frames: usize,
        completed: Arc<AtomicUsize>,
        completed_frames: Arc<AtomicUsize>,
        total_size: Arc<AtomicU64>,
    ) -> Self {
        let multi = MultiProgress::new();
        let style = msg_style();

        let mut worker_bars = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let bar = multi.add(ProgressBar::new_spinner());
            bar.set_style(style.clone());
            bar.set_message(" ");
            worker_bars.push(bar);
        }

        let summary_bar = multi.add(ProgressBar::new_spinner());
        summary_bar.set_style(style);

        let total_chunks = chunks.len();
        let total_frames = chunks.iter().map(|c| c.end - c.start).sum();

        let state = Arc::new(SummaryState {
            total_chunks,
            total_frames,
            fps_num: inf.fps_num as usize,
            fps_den: inf.fps_den as usize,
            completed,
            completed_frames,
            total_size,
            processed: AtomicUsize::new(0),
            start: Instant::now(),
            init_frames,
        });

        let stop = Arc::new(AtomicBool::new(false));

        let ticker = {
            let summary = summary_bar.clone();
            let tick_state = Arc::clone(&state);
            let tick_stop = Arc::clone(&stop);
            spawn(move || {
                summary_tick(&summary, &tick_state, &tick_stop);
            })
        };

        Self {
            multi,
            worker_bars,
            summary_bar,
            state,
            stop,
            ticker: Some(ticker),
        }
    }

    pub fn watch_enc<R: Read + Send + 'static>(
        &self,
        stderr: R,
        worker_id: usize,
        chunk_idx: usize,
        track_frames: bool,
        crf_score: Option<(f32, Option<f64>)>,
        encoder: Encoder,
    ) {
        let bar = self.worker_bars[worker_id].clone();
        let state = Arc::clone(&self.state);

        spawn(move || match encoder {
            SvtAv1 => {
                watch_svt(&bar, &state, stderr, chunk_idx, track_frames, crf_score);
            }
            Avm => {
                watch_avm(&bar, stderr, chunk_idx);
            }
            X265 | X264 => {
                watch_x265(&bar, &state, stderr, chunk_idx, track_frames, crf_score);
            }
            Vvenc => {
                watch_vvenc(&bar, &state, stderr, chunk_idx, track_frames, crf_score);
            }
        });
    }

    #[cfg(feature = "vship")]
    pub fn show_metric_progress(
        &self,
        worker_id: usize,
        chunk_idx: usize,
        progress: (usize, usize),
        fps: f32,
        crf_score: (f32, Option<f64>),
    ) {
        let (current, total) = progress;
        let (crf, last_score) = crf_score;
        let filled = (BAR_WIDTH * current / total.max(1)).min(BAR_WIDTH);
        let bar = format!(
            "{}{}",
            G_HASH.repeat(filled),
            R_DASH.repeat(BAR_WIDTH - filled)
        );
        let perc = (current * 100 / total.max(1)).min(100);
        let score_str = last_score.map_or(String::new(), |s| format!(" / {s:.2}"));

        let line = format!(
            "{C}[{chunk_idx:04} / F {crf:.2}{score_str}{C}] [{bar}{C}] {W}{perc:3}%{C}, \
             {Y}{fps:6.2}{C}, {G}{current:3}{C}/{R}{total}"
        );

        self.worker_bars[worker_id].set_message(line);
    }

    pub fn update_lib_enc(
        &self,
        worker_id: usize,
        chunk_idx: usize,
        progress: (usize, usize),
        fps: f32,
        frames_delta: Option<usize>,
        crf_score: Option<(f32, Option<f64>)>,
    ) {
        let (current, total) = progress;
        let filled = (BAR_WIDTH * current / total.max(1)).min(BAR_WIDTH);
        let bar = format!(
            "{}{}",
            B_HASH.repeat(filled),
            Y_DASH.repeat(BAR_WIDTH - filled)
        );
        let perc = (current * 100 / total.max(1)).min(100);

        let prefix = match crf_score {
            Some((crf, Some(score))) => {
                format!("{C}[{chunk_idx:04} / F {crf:.2} / {score:.2}{C}]")
            }
            Some((crf, None)) => format!("{C}[{chunk_idx:04} / F {crf:.2}{C}]"),
            None => format!("{C}[{chunk_idx:04}{C}]"),
        };

        let line = format!(
            "{prefix} {P}[{bar}{P}] {W}{perc:2}%{C}, {Y}{fps:6.2}{C}, {G}{current:3}{C}/{R}{total}"
        );

        if let Some(delta) = frames_delta {
            self.state.processed.fetch_add(delta, Ordering::Relaxed);
        }
        self.worker_bars[worker_id].set_message(line);
    }

    pub fn clear_lib_enc(&self, worker_id: usize) {
        self.worker_bars[worker_id].set_message(" ");
    }
}

impl Drop for ProgsTrack {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.ticker.take() {
            let _res = h.join();
        }
        for bar in &self.worker_bars {
            bar.finish_and_clear();
        }
        self.summary_bar.finish_and_clear();
    }
}

pub struct LibEncTracker {
    start: Instant,
    pub encoded: usize,
    last_reported: usize,
    pub worker_id: usize,
    chunk_idx: usize,
    total: usize,
    track_frames: bool,
    crf_score: Option<(f32, Option<f64>)>,
}

impl LibEncTracker {
    pub fn new(
        worker_id: usize,
        chunk_idx: usize,
        total: usize,
        track_frames: bool,
        crf_score: Option<(f32, Option<f64>)>,
    ) -> Self {
        Self {
            start: Instant::now(),
            encoded: 0,
            last_reported: 0,
            worker_id,
            chunk_idx,
            total,
            track_frames,
            crf_score,
        }
    }

    pub fn report(&mut self, prog: &ProgsTrack) {
        if self.encoded == self.last_reported {
            return;
        }
        let fps = self.encoded as f32 / self.start.elapsed().as_secs_f32().max(0.001);
        let delta = self.encoded - self.last_reported;
        self.last_reported = self.encoded;
        prog.update_lib_enc(
            self.worker_id,
            self.chunk_idx,
            (self.encoded, self.total),
            fps,
            self.track_frames.then_some(delta),
            self.crf_score,
        );
    }
}

fn summary_tick(bar: &ProgressBar, state: &SummaryState, stop: &AtomicBool) {
    while !stop.load(Ordering::Relaxed) {
        sleep(Duration::from_millis(INTERVAL_MS));
        update_summary(bar, state);
    }
    update_summary(bar, state);
}

fn update_summary(bar: &ProgressBar, state: &SummaryState) {
    let completed_frames = state.completed_frames.load(Ordering::Relaxed);
    let total_size = state.total_size.load(Ordering::Relaxed);
    let processed_frames = state.processed.load(Ordering::Relaxed);
    let frames_done = completed_frames.max(state.init_frames + processed_frames);

    let elapsed_secs =
        PRIOR_SECS.load(Ordering::Relaxed) as usize + state.start.elapsed().as_secs() as usize;
    let fps = frames_done as f32 / elapsed_secs.max(1) as f32;
    let remaining = state.total_frames.saturating_sub(frames_done);
    let eta_secs = remaining * elapsed_secs / frames_done.max(1);
    let chunks_done = state.completed.load(Ordering::Relaxed);

    let (bitrate_str, est_str) = if completed_frames > 0 {
        let dur = completed_frames as f32 * state.fps_den as f32 / state.fps_num as f32;
        let kbps = total_size as f32 * 8.0 / dur / 1000.0;
        let total_dur = state.total_frames as f32 * state.fps_den as f32 / state.fps_num as f32;
        let est_size = kbps * total_dur * 1000.0 / 8.0;
        let est = if est_size > 1_000_000_000.0 {
            format!("{:.1}g", est_size / 1_000_000_000.0)
        } else {
            format!("{:.1}m", est_size / 1_000_000.0)
        };
        (format!("{B}{kbps:.0}k"), format!("{R}{est}"))
    } else {
        (format!("{B}0k"), format!("{R}0m"))
    };

    let progress = (frames_done * BAR_WIDTH / state.total_frames.max(1)).min(BAR_WIDTH);
    let perc = (frames_done * 100 / state.total_frames.max(1)).min(100);
    let pbar = format!(
        "{}{}",
        G_HASH.repeat(progress),
        R_DASH.repeat(BAR_WIDTH - progress)
    );

    let (h, m) = (elapsed_secs / 3600, (elapsed_secs % 3600) / 60);
    let eta_h = (eta_secs / 3600).min(99);
    let eta_m = (eta_secs % 3600) / 60;

    bar.set_message(format!(
        "{W}{h:02}{P}:{W}{m:02} {C}[{G}{chunks_done}{C}/{R}{}{C}] [{pbar}{C}] {W}{perc}% \
         {G}{frames_done}{C}/{R}{} {C}({Y}{fps:.2}{C}, {W}{eta_h:02}{P}:{W}{eta_m:02}{C}, \
         {bitrate_str}{C}, {est_str}{C}{N})",
        state.total_chunks, state.total_frames
    ));
}

fn watch_svt(
    bar: &ProgressBar,
    state: &SummaryState,
    stderr: impl Read,
    chunk_idx: usize,
    track_frames: bool,
    crf_score: Option<(f32, Option<f64>)>,
) {
    let reader = BufReader::new(stderr);
    let mut last_frames = 0;

    for line in reader.split(b'\r').filter_map(Result::ok) {
        let Ok(text) = from_utf8(&line) else {
            continue;
        };
        let text = text.trim();

        if text.contains("error") || text.contains("Error") {
            eprint(format_args!("{text}"));
        }

        if text.is_empty() || !text.contains("Encoding:") || text.contains("SUMMARY") {
            continue;
        }

        let Some((current, total, fps, kbps)) = parse_svt(text) else {
            continue;
        };

        let prefix = match crf_score {
            Some((crf, Some(score))) => {
                format!("{C}[{chunk_idx:04} / F {crf:.2} / {score:.2}{C}]")
            }
            Some((crf, None)) => format!("{C}[{chunk_idx:04} / F {crf:.2}{C}]"),
            None => format!("{C}[{chunk_idx:04}{C}]"),
        };

        let filled = (BAR_WIDTH * current / total.max(1)).min(BAR_WIDTH);
        let pbar = format!(
            "{}{}",
            B_HASH.repeat(filled),
            Y_DASH.repeat(BAR_WIDTH - filled)
        );
        let perc = (current * 100 / total.max(1)).min(100);

        let display_line = format!(
            "{prefix} {P}[{pbar}{P}] {W}{perc:2}% {Y}{current:3}/{total} {G}{fps:6.2} {W}| \
             {P}{kbps:.0} kb/s"
        );

        if track_frames {
            let delta = current.saturating_sub(last_frames);
            last_frames = current;
            state.processed.fetch_add(delta, Ordering::Relaxed);
        }

        bar.set_message(display_line);
    }

    bar.set_message(" ");
}

fn watch_avm(bar: &ProgressBar, mut stdout: impl Read, chunk_idx: usize) {
    bar.set_message(format!(
        "{C}[{chunk_idx:04}]{W} Encoding: Progress updates when chunk finishes"
    ));

    let mut buf = [0u8; 4096];
    while stdout.read(&mut buf).unwrap_or(0) > 0 {}

    bar.set_message(" ");
}

fn watch_vvenc(
    bar: &ProgressBar,
    state: &SummaryState,
    mut stdout: impl Read,
    chunk_idx: usize,
    track_frames: bool,
    crf_score: Option<(f32, Option<f64>)>,
) {
    let start = Instant::now();
    let mut buf = [0u8; 4096];
    let mut line_buf = String::new();
    let mut poc_count = 0;
    let mut total_frames = 0;
    let mut last_poc_count = 0;
    let mut last_update = Instant::now();

    loop {
        match stdout.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                line_buf.push_str(&String::from_utf8_lossy(&buf[..n]));

                while let Some(pos) = line_buf.find('\n') {
                    let line = line_buf[..pos].trim().to_owned();
                    line_buf = line_buf[pos + 1..].to_string();

                    if line.contains("error") || line.contains("Error") {
                        eprint(format_args!("{line}"));
                    }

                    if total_frames == 0 && line.contains("encode ") {
                        total_frames = line
                            .split_whitespace()
                            .find_map(|s| s.parse().ok())
                            .unwrap_or(0);
                    }

                    if line.starts_with("POC") {
                        poc_count += 1;
                    }
                }

                if last_update.elapsed() >= Duration::from_millis(INTERVAL_MS) {
                    last_update = Instant::now();

                    let total = total_frames.max(poc_count);
                    let fps = poc_count as f32 / start.elapsed().as_secs_f32().max(0.001);
                    let filled = (BAR_WIDTH * poc_count / total.max(1)).min(BAR_WIDTH);
                    let pbar = format!(
                        "{}{}",
                        B_HASH.repeat(filled),
                        Y_DASH.repeat(BAR_WIDTH - filled)
                    );
                    let perc = (poc_count * 100 / total.max(1)).min(100);

                    let prefix = match crf_score {
                        Some((crf, Some(score))) => {
                            format!("{C}[{chunk_idx:04} / F {crf:.2} / {score:.2}{C}]")
                        }
                        Some((crf, None)) => format!("{C}[{chunk_idx:04} / F {crf:.2}{C}]"),
                        None => format!("{C}[{chunk_idx:04}{C}]"),
                    };

                    let display = format!(
                        "{prefix} {P}[{pbar}{P}] {W}{perc:2}%{C}, {Y}{fps:6.2}{C}, \
                         {G}{poc_count:3}{C}/{R}{total}"
                    );

                    if track_frames {
                        let d = poc_count.saturating_sub(last_poc_count);
                        last_poc_count = poc_count;
                        state.processed.fetch_add(d, Ordering::Relaxed);
                    }

                    bar.set_message(display);
                }
            }
        }
    }

    bar.set_message(" ");
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            i += 2;
            while i < bytes.len() && !bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
        } else {
            out.push(bytes[i] as char);
        }
        i += 1;
    }
    out
}

fn parse_svt(line: &str) -> Option<(usize, usize, f32, f32)> {
    let clean = strip_ansi(line);

    let frames_pos = clean.find(" Frames")?;
    let bytes = clean.as_bytes();

    let mut start = frames_pos;
    while start > 0 {
        let b = bytes[start - 1];
        if b.is_ascii_digit() || b == b'/' {
            start -= 1;
        } else {
            break;
        }
    }

    let num_part = &clean[start..frames_pos];
    let mut frame_parts = num_part.split('/');
    let current: usize = frame_parts.next()?.parse().ok()?;
    let total: usize = frame_parts.next()?.parse().ok()?;

    let after_frames = &clean[frames_pos + 7..];

    let fps = if let Some(fpm_pos) = after_frames.find(" fpm") {
        let before = &after_frames[..fpm_pos];
        let num_str = before
            .rsplit(|c: char| !c.is_ascii_digit() && c != '.')
            .next()?;
        num_str.parse::<f32>().ok()? / 60.0
    } else if let Some(fps_pos) = after_frames.find(" fps") {
        let before = &after_frames[..fps_pos];
        let num_str = before
            .rsplit(|c: char| !c.is_ascii_digit() && c != '.')
            .next()?;
        num_str.parse().ok()?
    } else {
        return None;
    };

    let kbps = if let Some(kbps_pos) = after_frames.find(" kb/s") {
        let before = &after_frames[..kbps_pos];
        let num_str = before
            .rsplit(|c: char| !c.is_ascii_digit() && c != '.')
            .next()?;
        num_str.parse().unwrap_or(0.0)
    } else {
        0.0
    };

    Some((current, total, fps, kbps))
}

fn watch_x265(
    bar: &ProgressBar,
    state: &SummaryState,
    stderr: impl Read,
    chunk_idx: usize,
    track_frames: bool,
    crf_score: Option<(f32, Option<f64>)>,
) {
    let reader = BufReader::new(stderr);
    let mut last_frames = 0;
    let mut last_update = Instant::now();

    for line in reader.split(b'\r').filter_map(Result::ok) {
        let Ok(text) = from_utf8(&line) else {
            continue;
        };
        let text = text.trim();

        if text.is_empty() {
            continue;
        }

        if !text.starts_with('[') {
            if !text.starts_with("encoded") {
                eprint(format_args!("{text}"));
            }
            continue;
        }

        if last_update.elapsed() < Duration::from_millis(INTERVAL_MS) {
            continue;
        }
        last_update = Instant::now();

        let Some((cur, tot, fps, kbps)) = parse_x265(text) else {
            continue;
        };

        let filled = (BAR_WIDTH * cur / tot.max(1)).min(BAR_WIDTH);
        let pbar = format!(
            "{}{}",
            B_HASH.repeat(filled),
            Y_DASH.repeat(BAR_WIDTH - filled)
        );

        let prefix = match crf_score {
            Some((crf, Some(s))) => format!("{C}[{chunk_idx:04} / F {crf:.2} / {s:.2}{C}]"),
            Some((crf, None)) => format!("{C}[{chunk_idx:04} / F {crf:.2}{C}]"),
            None => format!("{C}[{chunk_idx:04}{C}]"),
        };

        let line = format!(
            "{prefix} {P}[{pbar}{P}] {W}{:2}% {Y}{cur:3}/{tot} {G}{fps:6.2} {W}| {P}{kbps:.0} kb/s",
            cur * 100 / tot.max(1)
        );

        if track_frames {
            let d = cur.saturating_sub(last_frames);
            last_frames = cur;
            state.processed.fetch_add(d, Ordering::Relaxed);
        }

        bar.set_message(line);
    }

    bar.set_message(" ");
}

fn parse_x265(s: &str) -> Option<(usize, usize, f32, f32)> {
    let rest = s.split(']').nth(1)?;
    let mut parts = rest.split(',');

    let fp = parts.next()?.trim();
    let mut fs = fp.split('/');
    let cur = fs.next()?.trim().parse().ok()?;
    let tot = fs.next()?.split_whitespace().next()?.parse().ok()?;

    let fps = parts.next()?.split_whitespace().next()?.parse().ok()?;
    let kbps = parts.next()?.split_whitespace().next()?.parse().ok()?;

    Some((cur, tot, fps, kbps))
}
