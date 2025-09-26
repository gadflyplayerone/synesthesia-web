
// src/visuals/Boids.ts
export type Boid = { x:number; y:number; vx:number; vy:number; hue:number };
export type BoidParams = {
  count: number;
  maxSpeed: number;
  align: number; // alignment weight
  coh: number;   // cohesion weight
  sep: number;   // separation weight
  range: number; // neighbor range
};

export function makeBoids(W:number,H:number, count=120): Boid[] {
  const arr: Boid[] = [];
  for (let i=0;i<count;i++) {
    arr.push({
      x: Math.random()*W,
      y: Math.random()*H,
      vx: (Math.random()*2-1)*0.5,
      vy: (Math.random()*2-1)*0.5,
      hue: Math.random()*360
    });
  }
  return arr;
}

export function stepBoids(boids:Boid[], W:number, H:number, params:BoidParams, target?: {x:number;y:number}, bassPush=0){
  const {maxSpeed, align, coh, sep, range} = params;
  for (let i=0;i<boids.length;i++){
    const b = boids[i];
    let vx=0, vy=0, ax=0, ay=0, sx=0, sy=0, cx=0, cy=0;
    let n=0;

    for (let j=0;j<boids.length;j++){
      if (i===j) continue;
      const o = boids[j];
      const dx = o.x - b.x, dy = o.y - b.y;
      const d2 = dx*dx+dy*dy;
      if (d2 < range*range){
        n++;
        // alignment
        ax += o.vx; ay += o.vy;
        // cohesion
        cx += o.x; cy += o.y;
        // separation
        const inv = Math.max(1e-3, d2);
        sx -= dx / inv; sy -= dy / inv;
      }
    }
    if (n>0){
      ax = (ax/n - b.vx) * align;
      ay = (ay/n - b.vy) * align;
      cx = ((cx/n) - b.x) * coh;
      cy = ((cy/n) - b.y) * coh;
      sx = sx * sep;
      sy = sy * sep;
    }
    // target attraction
    if (target){
      const dx = target.x - b.x, dy = target.y - b.y;
      vx += dx*0.0008; vy += dy*0.0008;
    }

    b.vx += ax + sx + cx + vx;
    b.vy += ay + sy + cy + vy;

    // bass energy push outward
    b.vx += (b.x - W/2) * 0.00002 * bassPush;
    b.vy += (b.y - H/2) * 0.00002 * bassPush;

    // clamp speed
    const sp = Math.hypot(b.vx,b.vy);
    if (sp > maxSpeed){
      b.vx = b.vx / sp * maxSpeed;
      b.vy = b.vy / sp * maxSpeed;
    }

    b.x += b.vx; b.y += b.vy;
    if (b.x<0) b.x+=W; if (b.x>W) b.x-=W;
    if (b.y<0) b.y+=H; if (b.y>H) b.y-=H;
  }
}

// color drift
export function drawBoids(ctx:CanvasRenderingContext2D, boids:Boid[]){
  ctx.save();
  for (const b of boids){
    ctx.fillStyle = `hsla(${b.hue}, 90%, 70%, 0.9)`;
    ctx.beginPath();
    ctx.arc(b.x, b.y, 2.2, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.restore();
}
