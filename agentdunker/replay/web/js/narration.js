// 语音解说：每条字幕对应 audio/<play.id>/NN.mp3（manifest.json 列出）；没有音频文件时退回浏览器 speechSynthesis。
// main.js 用 busy() 判断"这句还没说完"，在字幕末尾定格等待，保证画面和解说同步。
export class Narrator {
  constructor(play) {
    this.play = play;
    this.base = `./audio/${play.id}/`;
    this.clips = null;
    this.enabled = false;
    this.idx = -1;
    this.audio = new Audio();
    this.audio.preload = 'auto';
    this.ctx = null;
    this.dest = null;
    this.utter = null;
    this.ready = play.id
      ? fetch(`${this.base}manifest.json`).then((r) => (r.ok ? r.json() : null)).then((m) => { this.clips = m?.clips ?? null; }).catch(() => {})
      : Promise.resolve();
  }

  get available() {
    return !!(this.play.captions?.length && (this.clips || 'speechSynthesis' in window));
  }

  /** 必须在用户点击里调用（浏览器自动播放策略）。同时把音频接到录制用的 MediaStream。 */
  enable() {
    this.enabled = true;
    if (!this.ctx && window.AudioContext) {
      this.ctx = new AudioContext();
      const src = this.ctx.createMediaElementSource(this.audio);
      this.dest = this.ctx.createMediaStreamDestination();
      src.connect(this.ctx.destination);
      src.connect(this.dest);
    }
    this.ctx?.resume();
  }

  disable() {
    this.stop();
    this.enabled = false;
  }

  speak(i, rate = 1) {
    this.stop();
    this.idx = i;
    if (!this.enabled) return;
    const clip = this.clips?.[i];
    if (clip) {
      this.audio.src = this.base + clip.file;
      this.audio.playbackRate = rate;
      this.audio.play().catch(() => {});
    } else if ('speechSynthesis' in window) {
      this.utter = new SpeechSynthesisUtterance(this.play.captions[i][2]);
      this.utter.lang = 'zh-CN';
      this.utter.rate = 1.1 * rate;
      this.utter.onend = () => { this.utter = null; };
      speechSynthesis.speak(this.utter);
    }
  }

  busy() {
    return (!this.audio.paused && !this.audio.ended) || !!this.utter;
  }

  pause() {
    this.audio.pause();
    if (this.utter) speechSynthesis.pause();
  }

  resume() {
    if (!this.enabled || this.idx < 0) return;
    if (this.audio.src && !this.audio.ended && this.audio.currentTime > 0) this.audio.play().catch(() => {});
    if (this.utter) speechSynthesis.resume();
  }

  stop() {
    this.audio.pause();
    if (this.audio.src) this.audio.removeAttribute('src');
    if (this.utter || ('speechSynthesis' in window && speechSynthesis.speaking)) speechSynthesis.cancel();
    this.utter = null;
    this.idx = -1;
  }

  /** 录制用：解说音轨（speechSynthesis 无法录入）。 */
  audioTracks() {
    return this.dest ? this.dest.stream.getAudioTracks() : [];
  }
}
