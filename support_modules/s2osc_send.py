import time
import socket
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

IP = '127.0.0.1'
PORT = 7474

client = udp_client.UDPClient(IP, PORT)

full_msg = b''


def osc_send(_msg):
    
    o_msg = OscMessageBuilder(address='/fromYolo')
    o_msg.add_arg(_msg)
    m = o_msg.build()
    client.send(m)

while True:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((socket.gethostname(), 7010))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        msg = s.recv(1024)
        print('Original Message = ' + str(msg))
        numbered_msg = int.from_bytes(msg, byteorder='big')
        use_msg = '0.' + str(numbered_msg)
        # numbered_msg = int(msg)
        print('Got Message =' + use_msg)
        #run(CONFIG_MAP, float(use_msg))
        osc_send(use_msg)
    except ConnectionRefusedError:
        msg = 'Waiting...'
        print(msg)
        time.sleep(5)
        continue