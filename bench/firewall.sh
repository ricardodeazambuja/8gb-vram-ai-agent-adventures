#!/bin/sh -eu
# firewall.sh — keep benchmark containers off the LAN and the host (needs root).
#
#   sh bench/firewall.sh add      # install the rules (idempotent)
#   sh bench/firewall.sh remove   # take them out again
#
# Containers on the pibench network (bridge br-pibench, 172.30.0.0/24) may reach the internet
# and the model server on the host's bridge address, 172.30.0.1:8080. Nothing else:
#   * DOCKER-USER (forwarded traffic): drop anything to private, link-local and CGNAT ranges,
#     so the LAN, the router and other Docker networks are unreachable.
#   * INPUT (traffic to the host itself, on any of its addresses): accept TCP 8080 from the
#     bridge, drop the rest.
# The rules are not saved: they disappear on reboot or when Docker re-creates its chains.
# DNS still works: Docker's resolver for user-defined networks runs inside the container.

BR=br-pibench
ACTION="${1:-add}"

rule() {  # rule CHAIN ARGS... — add (once) or delete one rule
  chain="$1"; shift
  if [ "$ACTION" = add ]; then
    iptables -C "$chain" "$@" 2>/dev/null || iptables -I "$chain" 1 "$@"
  else
    while iptables -C "$chain" "$@" 2>/dev/null; do iptables -D "$chain" "$@"; done
  fi
}

case "$ACTION" in add|remove) ;; *) echo "usage: $0 add|remove" >&2; exit 2 ;; esac

for net in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 169.254.0.0/16 100.64.0.0/10; do
  rule DOCKER-USER -i "$BR" -d "$net" -j DROP
done
# Inserted in this order so the ACCEPT ends up above the DROP.
rule INPUT -i "$BR" -j DROP
rule INPUT -i "$BR" -p tcp --dport 8080 -j ACCEPT

echo "== DOCKER-USER"; iptables -S DOCKER-USER | grep -- "$BR" || true
echo "== INPUT";       iptables -S INPUT | grep -- "$BR" || true
