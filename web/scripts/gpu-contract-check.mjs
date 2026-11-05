import { readFileSync, existsSync } from 'node:fs';

const files = ['src/lib/gpu-api.ts', 'src/components/GpuParamForm.tsx', 'src/App.tsx', 'src/styles.css'];
for (const file of files) if (!existsSync(file)) throw new Error(`missing standalone GPU file: ${file}`);
const api = readFileSync('src/lib/gpu-api.ts', 'utf8');
for (const symbol of ['unwrapGpuResponse', 'createGpuJob', 'getGpuWorkflowSchema', 'Idempotency-Key', 'login']) {
  if (!api.includes(symbol)) throw new Error(`missing GPU contract symbol: ${symbol}`);
}
const app = readFileSync('src/App.tsx', 'utf8');
for (const markers of [['สร้างสรรค์', '创作'], ['งาน', '任务'], ['ผลงาน', '作品'], ['เวิร์กโฟลว์', '工作流'], ['ตั้งค่า', '设置'], ['role="dialog"']]) {
  if (!markers.some((marker) => app.includes(marker))) {
    throw new Error(`missing business UI marker: ${markers.join(' or ')}`);
  }
}
console.log('standalone GPU contract markers present');
