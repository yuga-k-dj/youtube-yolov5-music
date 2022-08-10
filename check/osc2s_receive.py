import socket

from pythonosc import dispatcher
from pythonosc import osc_server

OSC_IP = '127.0.0.1'
# OSC_IP = '10.0.1.33'
OSC_PORT = 8080

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 7010))

def socket_send(message):
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind((socket.gethostname(), 7010))
    s.listen(8)
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established!")
    clientsocket.send(bytes(str(message), 'utf-8'))
    # clientsocket.send(str(message), 'utf-8')
    clientsocket.close()



def print4socket(unused_addr, args, volume):
  try:
    # print("[{0}] ~ {1}".format(args[0], args[1](volume)))
    # print(args)
    # print(volume)
    if volume == '0.49':
      print('Fail because: ' + volume)
      pass
    else:
      print(volume)
      socket_send(volume)
  except ValueError: pass

if __name__ == "__main__":

  dispatcher = dispatcher.Dispatcher()
  dispatcher.map("/fromMax", print4socket, "Volume")

  server = osc_server.ThreadingOSCUDPServer(
      (OSC_IP, OSC_PORT), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()