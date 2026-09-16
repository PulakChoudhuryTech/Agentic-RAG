---
type: Runbook
tags: [it, vpn, network]
related: [it/password_reset_policy.md]
stale_after: 2027-01-01
---
# VPN Troubleshooting Guide

**Category:** IT | **Applies to:** All employees | **Effective date:** 2025-01-01

## Overview

This guide covers common VPN connectivity issues and step-by-step fixes.
Acme Global uses GlobalProtect VPN client for all remote access to internal
systems.

## Common Issue: VPN Won't Connect

If the VPN client shows "Unable to connect" or hangs on "Connecting...":

1. **Check your internet connection.** Open a browser and confirm you can
   load a public website. If not, the issue is your local network, not VPN.
2. **Restart the GlobalProtect client.** Right-click the GlobalProtect icon in
   your system tray/menu bar and select "Disconnect", wait 10 seconds, then
   "Connect".
3. **Verify the portal address.** It should be `vpn.acmeglobal.com`. If it has
   changed or is blank, re-enter it in Settings.
4. **Restart your machine.** This resolves most stale-session issues,
   especially after a Windows/macOS update.
5. **Clear the client cache.** On Windows: delete the contents of
   `C:\ProgramData\Palo Alto Networks\GlobalProtect\`. On macOS: delete
   `~/Library/Application Support/GlobalProtect/`. Then restart the client.
6. **Check for a known outage.** Visit the IT Status page
   (status.acmeglobal.com) for any active incidents affecting VPN.

## Common Issue: VPN Connects but No Internal Access

If VPN shows "Connected" but internal sites (Workday, Confluence, internal
Git) don't load:

1. Confirm the VPN client shows a green "Connected" status, not "Connecting"
   or a warning icon.
2. Run `ipconfig /all` (Windows) or `ifconfig` (macOS/Linux) and confirm you
   have received a `10.x.x.x` internal IP address. If you still have only your
   local network IP, the tunnel did not fully establish -- disconnect and
   reconnect.
3. Flush your DNS cache: `ipconfig /flushdns` (Windows) or
   `sudo dscacheutil -flushcache` (macOS).
4. Try an internal resource by IP address instead of hostname to rule out
   internal DNS issues.

## Common Issue: Repeated Authentication Prompts

1. Confirm your corporate password hasn't expired (see Password Reset
   Policy).
2. If using MFA, ensure your device clock is synced (incorrect time causes
   TOTP codes to fail) -- enable "Set time automatically" in your OS settings.
3. If prompts persist after a password reset and correct MFA codes, this
   usually indicates an account lockout -- proceed to escalation below.

## When to Escalate to a Ticket

If you have completed the steps above (client restart, machine restart, cache
clear, DNS flush) and VPN still will not connect or will not grant internal
access, this is not self-serviceable and should be escalated by creating an
IT support ticket in ServiceNow with:

- Your operating system and GlobalProtect client version
- Which step above failed and any error message shown
- Whether the issue is intermittent or constant

Category for the ticket: "Network > VPN". Typical resolution time for VPN
tickets is 4 business hours.
