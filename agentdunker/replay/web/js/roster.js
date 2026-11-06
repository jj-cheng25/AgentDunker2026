// 球员姓名占位：战术脚本里的文字用 {球员id} 引用球员，加载时替换成 roster 里的姓名。
// 姓名留空时用占位：号码牌显示队名，字幕/解说显示「队名 + 号码号」（如「火箭7号」）。
export function paletteLabels(play) {
  if (play.metadata?.teamRolesVerified !== false) return {off:'进攻',def:'防守'};
  const neutral = {off:'青色阵营',def:'橙色阵营'};
  if (play.metadata?.teamMembershipVerified !== true) return neutral;
  return Object.fromEntries(['off','def'].map(key=>{
    const name=play.teams?.[key];
    const safe=typeof name==='string' && name.trim() &&
      !/进攻|防守|offen|defen|attack|^off$|^def$/i.test(name);
    return [key,safe?`${name} · ${key==='off'?'青色':'橙色'}`:neutral[key]];
  }));
}

export function resolveNames(play, roster = {}) {
  const teamName = (p) => play.metadata?.teamRolesVerified === false
    ? paletteLabels(play)[p.team] : play.teams?.[p.team] ?? (p.team === 'off' ? '进攻' : '防守');
  for (const p of play.players) {
    if (p.identityVerified === false || p.identityKnown === false) {
      // Body membership has its own source proof. It never grants a name or
      // jersey, including through a stale roster or a caller's shortName.
      p.player = p.name = p.say = '球员';
      p.tag = p.shortName = '未确认姓名';
      p.identityVerified = p.identityKnown = p.nameVerified = p.jerseyVerified = false;
      p.identityConfidence = 0;
      p.num = '';
      delete p.playerId; delete p.jersey;
      if (roster && typeof roster === 'object') roster[p.id] = {name:'',shortName:'',num:''};
      if (play.roster && play.roster !== roster) play.roster[p.id] = {name:'',shortName:'',num:''};
      continue;
    }
    const r = roster[p.id] ?? { name: p.player ?? '', num: p.num };
    if (r.num) p.num = r.num;
    p.player = r.name || '';
    const short = r.shortName || p.shortName || (r.name ? r.name.split(/[·•]/).at(-1) : '');
    p.tag = short || teamName(p);
    p.say = short || (p.num ? `${teamName(p)}${p.num}号` : `${teamName(p)}球员`);
  }
  const byId = Object.fromEntries(play.players.map((p) => [p.id, p.say]));
  const sub = (s) => (typeof s === 'string' ? s.replace(/\{(\w+)\}/g, (m, id) => byId[id] ?? m) : s);
  play.title = sub(play.title);
  play.captions = (play.captions ?? []).map(([a, b, s]) => [a, b, sub(s)]);
  if (play.exportCaptions) play.exportCaptions=play.exportCaptions.map(([a,b,s])=>[a,b,sub(s)]);
  for (const a of play.annotations ?? []) { a.label = sub(a.label); a.text = sub(a.text); }
  for (const ph of play.phases ?? []) { ph.name = sub(ph.name); ph.intent = sub(ph.intent); }
  return play;
}
