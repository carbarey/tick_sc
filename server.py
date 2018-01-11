import json
from http.server import *
import ticket_nn


class GetHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        try:
            
            if self.path.endswith("/test"):

                print("\n***\n")
                content_len = int(self.headers['Content-Length'])
                post_body = self.rfile.read(content_len)

                data = str(post_body.decode('utf-8'))
                json_data = json.loads(data)
                
                print(json_data)
            
                c = json_data["c"]
                i = json_data["i"]
                s = json_data["s"]
                o = json_data["o"]
                n = json_data["n"]
                n_preds = int(json_data["n_preds"])

                predicted_groups = ticket_nn.predict_group(c,i,s,o,n, n_preds)
                print(predicted_groups)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                json_response = json.dumps({"groups": predicted_groups});
                self.wfile.write(bytes(json_response, 'utf-8'))

        except Exception as e:
            print("[ERROR]" + str(e))

        return


if __name__ == "__main__":
    HandlerClass = GetHandler
    ServerClass = HTTPServer

    protocol = "HTTP/1.0"
    host = "0.0.0.0"
    port = 5050

    server_address = (host, port)

    HandlerClass.protocol_version = protocol
    httpd = ServerClass(server_address, HandlerClass)
    print ('\n(listening at http://0.0.0.0:5050)')

    httpd.serve_forever()
    