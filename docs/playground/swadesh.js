// Tiny loader for swadesh_wasm.wasm (no dependencies). Works in browsers and Node.
export async function loadSwadesh(wasmBytesOrUrl) {
  const bytes = typeof wasmBytesOrUrl === "string"
    ? await (await fetch(wasmBytesOrUrl)).arrayBuffer() : wasmBytesOrUrl;
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const x = instance.exports;
  const enc = new TextEncoder(), dec = new TextDecoder();
  const put = (u8) => { const p = x.sw_alloc(u8.length); new Uint8Array(x.memory.buffer, p, u8.length).set(u8); return [p, u8.length]; };
  const read = (p, n) => dec.decode(new Uint8Array(x.memory.buffer, p, n).slice());
  return {
    /** Run the sacreBLEU-identical CLI. files: {path: string}; "<stdin>" is stdin. */
    run(args, files = {}) {
      x.sw_clear_files();
      for (const [name, text] of Object.entries(files)) {
        const [np, nl] = put(enc.encode(name)); const [dp, dl] = put(enc.encode(text));
        x.sw_set_file(np, nl, dp, dl); x.sw_free(np, nl); x.sw_free(dp, dl);
      }
      const [ap, al] = put(enc.encode(args.join("\0")));
      const code = x.sw_run(ap, al); x.sw_free(ap, al);
      return { code, stdout: read(x.sw_out_ptr(), x.sw_out_len()), stderr: read(x.sw_err_ptr(), x.sw_err_len()) };
    },
  };
}
