#!/usr/bin/env python3
"""
manage_keys.py - CLI tool for API key provisioning.

Usage
-----
  # Create a key for a given user
  python manage_keys.py create <user_id>

  # List all keys (no secrets shown)
  python manage_keys.py list

  # Revoke by prefix (first 8 chars of the raw key)
  python manage_keys.py revoke <key_prefix>

Examples
--------
  python manage_keys.py create unisinos
  python manage_keys.py create malta_internal
  python manage_keys.py list
  python manage_keys.py revoke malta_Xk
"""

import argparse
import sys

import db as db
import auth as auth

db.init_db()


def cmd_create(args):
    raw_key = auth.create_key(args.user_id)
    print("=" * 60)
    print(f"  User   : {args.user_id}")
    print(f"  Prefix : {raw_key[:auth.PREFIX_LEN]}")
    print(f"  Key    : {raw_key}")
    print()
    print("  ⚠  Store this key now – it will NOT be shown again.")
    print("=" * 60)


def cmd_list(args):
    rows = db.list_keys()
    if not rows:
        print("No API keys found.")
        return

    header = f"{'ID':>4}  {'USER':<20}  {'PREFIX':<10}  {'ACTIVE':<6}  {'CREATED':<27}  LAST USED"
    print(header)
    print("-" * len(header))
    for row in rows:
        last_used = row["last_used_at"] or "—"
        active = "yes" if row["active"] else "no"
        print(
            f"{row['id']:>4}  {row['user_id']:<20}  {row['key_prefix']:<10}  "
            f"{active:<6}  {row['created_at']:<27}  {last_used}"
        )


def cmd_revoke(args):
    count = auth.revoke_key(args.key_prefix)
    if count:
        print(f"Revoked {count} key(s) with prefix '{args.key_prefix}'.")
    else:
        print(f"No active keys found with prefix '{args.key_prefix}'.")


def main():
    parser = argparse.ArgumentParser(
        description="MALTA-GEO API key management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Generate a new API key")
    p_create.add_argument("user_id", help="Owner label, e.g. 'unisinos'")
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="List all keys (no secrets)")
    p_list.set_defaults(func=cmd_list)

    p_revoke = sub.add_parser("revoke", help="Revoke keys by prefix")
    p_revoke.add_argument("key_prefix", help="First 8 chars of the raw key")
    p_revoke.set_defaults(func=cmd_revoke)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()