import http from "node:http";

const host = process.env.VAIR_PUBLIC_PROXY_HOST ?? "127.0.0.1";
const port = Number.parseInt(process.env.VAIR_PUBLIC_PROXY_PORT ?? "8080", 10);
const uiOrigin = new URL(process.env.VAIR_UI_ORIGIN ?? "http://127.0.0.1:3200");
const apiOrigin = new URL(process.env.VAIR_API_ORIGIN ?? "http://127.0.0.1:8200");

if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error(`Invalid VAIR_PUBLIC_PROXY_PORT: ${process.env.VAIR_PUBLIC_PROXY_PORT}`);
}

function upstreamFor(pathname) {
  return pathname === "/api" || pathname.startsWith("/api/")
    ? apiOrigin
    : uiOrigin;
}

const server = http.createServer((request, response) => {
  const requestUrl = request.url ?? "/";
  const pathname = new URL(requestUrl, "http://proxy.local").pathname;
  const upstream = upstreamFor(pathname);
  const headers = {
    ...request.headers,
    host: upstream.host,
    "x-forwarded-host": request.headers.host ?? "",
  };

  const proxyRequest = http.request(
    {
      protocol: upstream.protocol,
      hostname: upstream.hostname,
      port: upstream.port,
      method: request.method,
      path: requestUrl,
      headers,
    },
    (proxyResponse) => {
      response.writeHead(
        proxyResponse.statusCode ?? 502,
        proxyResponse.statusMessage,
        proxyResponse.headers,
      );
      proxyResponse.pipe(response);
    },
  );

  proxyRequest.on("error", (error) => {
    if (!response.headersSent) {
      response.writeHead(502, { "content-type": "application/json" });
    }
    response.end(JSON.stringify({ detail: `Upstream unavailable: ${error.message}` }));
  });

  request.pipe(proxyRequest);
});

server.on("clientError", (_error, socket) => {
  socket.end("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n");
});

server.listen(port, host, () => {
  console.log(`VAIR public proxy listening on http://${host}:${port}`);
  console.log(`UI upstream: ${uiOrigin}`);
  console.log(`API upstream: ${apiOrigin}`);
});
