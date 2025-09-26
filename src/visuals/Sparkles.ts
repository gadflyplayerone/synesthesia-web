
// src/visuals/Sparkles.ts
export type Sparkle = { x:number;y:number;vx:number;vy:number;life:number;size:number };
export function burstSparkles(arr:Sparkle[], x:number,y:number, amount:number, energy:number){
  for (let i=0;i<amount;i++){
    const a = Math.random()*Math.PI*2;
    const s = (Math.random()*1.5+0.5) * (0.5+energy);
    arr.push({ x, y, vx: Math.cos(a)*s, vy: Math.sin(a)*s, life: 1.0, size: Math.random()*2+1 });
  }
}
export function stepSparkles(arr:Sparkle[], W:number,H:number){
  for (let i=arr.length-1; i>=0; i--){
    const p = arr[i];
    p.x += p.vx; p.y += p.vy;
    p.vx *= 0.98; p.vy *= 0.98;
    p.life -= 0.02;
    if (p.life <= 0 || p.x<0||p.x>W||p.y<0||p.y>H) arr.splice(i,1);
  }
}
export function drawSparkles(ctx:CanvasRenderingContext2D, arr:Sparkle[]){
  ctx.save();
  for (const p of arr){
    ctx.globalAlpha = Math.max(0, p.life);
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.restore();
}
