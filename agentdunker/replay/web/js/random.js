let seed = 0xC07A2026;
export function random(){ seed = (Math.imul(1664525,seed)+1013904223) >>> 0; return seed/4294967296; }
