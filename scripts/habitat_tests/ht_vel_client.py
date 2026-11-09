"""CLI client for ht_vel_server.py: sends velocity/record/save/stop commands
over UDP.

Usage:
    python scripts/habitat_tests/ht_vel_client.py vel <linear_x> <angular_y> [--rate hz] [--duration s]
"""

import argparse
import socket
import time

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5055


def send_command(host, port, command):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(command.encode("utf-8"), (host, port))
    sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="Send ht_vel commands to Habitat velocity server."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = parser.add_subparsers(dest="cmd", required=True)
    vel = sub.add_parser("vel")
    vel.add_argument("linear_x", type=float)
    vel.add_argument("angular_y", type=float)
    vel.add_argument("--rate", type=float, default=10.0)
    vel.add_argument("--duration", type=float, default=0.0)
    sub.add_parser("stop")
    rec = sub.add_parser("rec")
    rec.add_argument("mode", choices=["on", "off", "toggle"])
    sub.add_parser("save")
    clearance = sub.add_parser("clearance")
    clearance.add_argument("meters", type=float)
    sub.add_parser("quit")

    args = parser.parse_args()

    if args.cmd == "vel":
        command = f"ht_vel {args.linear_x} {args.angular_y}"
        if args.duration <= 0:
            send_command(args.host, args.port, command)
            print(command)
        else:
            dt = 1.0 / args.rate
            end = time.time() + args.duration

            while time.time() < end:
                send_command(args.host, args.port, command)
                time.sleep(dt)

            send_command(args.host, args.port, "ht_stop")
            print(command)
            print("ht_stop")

    elif args.cmd == "stop":
        send_command(args.host, args.port, "ht_stop")
        print("ht_stop")

    elif args.cmd == "rec":
        command = f"ht_rec {args.mode}"
        send_command(args.host, args.port, command)
        print(command)

    elif args.cmd == "save":
        send_command(args.host, args.port, "ht_save")
        print("ht_save")

    elif args.cmd == "clearance":
        command = f"ht_clearance {args.meters}"
        send_command(args.host, args.port, command)
        print(command)

    elif args.cmd == "quit":
        send_command(args.host, args.port, "ht_quit")
        print("ht_quit")


if __name__ == "__main__":
    main()
