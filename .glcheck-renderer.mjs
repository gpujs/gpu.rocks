import { launch } from './scripts/browser.mjs';
const browser = await launch();
const page = await browser.newPage();
const info = await page.evaluate(() => {
  const r = {};
  for (const [k, v] of [['webgl','webgl'],['webgl2','webgl2']]) {
    const c = document.createElement('canvas').getContext(v);
    if (!c) { r[k] = 'none'; continue; }
    const d = c.getExtension('WEBGL_debug_renderer_info');
    r[k] = d ? c.getParameter(d.UNMASKED_RENDERER_WEBGL) : c.getParameter(c.RENDERER);
    r[k+'_precision'] = (() => { const p = c.getShaderPrecisionFormat(c.FRAGMENT_SHADER, c.HIGH_FLOAT); return p ? [p.rangeMin,p.rangeMax,p.precision] : null; })();
    r[k+'_maxtex'] = c.getParameter(c.MAX_TEXTURE_SIZE);
    r[k+'_float'] = !!(c.getExtension('OES_texture_float') || v==='webgl2');
    r[k+'_cbf'] = !!(c.getExtension('WEBGL_color_buffer_float') || c.getExtension('EXT_color_buffer_float'));
  }
  return r;
});
console.log(JSON.stringify(info, null, 2));
await browser.close();
