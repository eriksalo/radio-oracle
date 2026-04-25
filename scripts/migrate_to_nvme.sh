#!/usr/bin/env bash
# Migrate the Jetson Orin Nano OS from SD card to NVMe SSD.
#
# Performs an in-place rootfs clone to /dev/nvme0n1 and reorders UEFI BootOrder
# so the NVMe boots first, with the original SD card as automatic fallback.
#
# Layout created on /dev/nvme0n1:
#   p1: 128MiB FAT32 ESP (boot/esp flags) — holds /EFI/BOOT/BOOTAA64.efi
#   p2: rest, ext4, PARTLABEL=APP — rootfs (L4TLauncher scans for this label)
#
# Usage: sudo ./scripts/migrate_to_nvme.sh [--dry-run]
#
# Prerequisites:
#   - Jetson Orin Nano running JetPack 6 (R36.x), currently booted from SD
#   - /dev/nvme0n1 present and not holding data you want to keep
#   - Physical access in case the first NVMe boot fails (UEFI falls back to SD automatically)

set -euo pipefail

DRY_RUN=false
NVME_DEV="/dev/nvme0n1"
ESP_PART="${NVME_DEV}p1"
APP_PART="${NVME_DEV}p2"
MOUNT="/mnt/newroot"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Would perform the following:"
fi

run() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "  $ $*"
    else
        "$@"
    fi
}

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root (sudo)." >&2
    exit 1
fi

if [[ ! -b "$NVME_DEV" ]]; then
    echo "$NVME_DEV not present." >&2
    exit 1
fi

echo "=== 1. Tear down anything on $NVME_DEV ==="
if vgs --noheadings 2>/dev/null | awk '{print $1}' | grep -q .; then
    for vg in $(vgs --noheadings -o vg_name | awk '{print $1}'); do
        run vgchange -an "$vg" || true
        run vgremove -f "$vg" || true
    done
fi
run pvremove -ffy "$NVME_DEV" 2>/dev/null || true
run wipefs -af "$NVME_DEV"

echo "=== 2. Partition (GPT: ESP 128MiB + APP rest) ==="
run parted -s "$NVME_DEV" mklabel gpt
run parted -s -a optimal "$NVME_DEV" mkpart esp fat32 1MiB 129MiB
run parted -s -a optimal "$NVME_DEV" set 1 esp on
run parted -s -a optimal "$NVME_DEV" mkpart APP ext4 129MiB 100%
run partprobe "$NVME_DEV"
run sleep 2

echo "=== 3. Format ==="
run mkfs.fat -F 32 -n EFI "$ESP_PART"
run mkfs.ext4 -L APP -F "$APP_PART"

echo "=== 4. Mount APP and rsync rootfs ==="
run mkdir -p "$MOUNT"
run mount "$APP_PART" "$MOUNT"
run rsync -aAXH --info=progress2 \
    --exclude="/proc/*" \
    --exclude="/sys/*" \
    --exclude="/dev/*" \
    --exclude="/run/*" \
    --exclude="/tmp/*" \
    --exclude="/mnt/*" \
    --exclude="/media/*" \
    --exclude="/lost+found" \
    --exclude="/boot/efi/*" \
    --exclude="/swapfile" \
    --exclude="/var/cache/apt/archives/*.deb" \
    --exclude="/var/lib/apt/lists/*" \
    --exclude="/var/log/journal/*" \
    / "$MOUNT/"

echo "=== 5. Populate new ESP with L4TLauncher ==="
run mkdir -p "$MOUNT/boot/efi"
run mount "$ESP_PART" "$MOUNT/boot/efi"
run cp -a /boot/efi/. "$MOUNT/boot/efi/"

echo "=== 6. Update fstab and extlinux.conf on the new disk ==="
NEW_APP_PARTUUID=$(blkid -s PARTUUID -o value "$APP_PART")
NEW_ESP_UUID=$(blkid -s UUID -o value "$ESP_PART")
OLD_ESP_UUID=$(blkid -s UUID -o value /dev/mmcblk0p10)

if [[ "$DRY_RUN" == true ]]; then
    echo "  Would replace UUID=$OLD_ESP_UUID with UUID=$NEW_ESP_UUID in $MOUNT/etc/fstab"
    echo "  Would replace root=/dev/mmcblk0p1 with root=PARTUUID=$NEW_APP_PARTUUID in $MOUNT/boot/extlinux/extlinux.conf"
else
    sed -i "s|UUID=${OLD_ESP_UUID}|UUID=${NEW_ESP_UUID}|g" "$MOUNT/etc/fstab"
    sed -i "s|root=/dev/mmcblk0p1|root=PARTUUID=${NEW_APP_PARTUUID}|g" "$MOUNT/boot/extlinux/extlinux.conf"
fi

echo "=== 7. Unmount ==="
run sync
run umount "$MOUNT/boot/efi"
run umount "$MOUNT"

echo "=== 8. Reorder UEFI BootOrder (NVMe first, SD as fallback) ==="
NVME_BOOT_NUM=$(efibootmgr | awk '/NVMe SSD/ {gsub(/Boot|\*/,"",$1); print $1; exit}')
SD_BOOT_NUM=$(efibootmgr | awk '/UEFI SD Device/ {gsub(/Boot|\*/,"",$1); print $1; exit}')

if [[ -z "$NVME_BOOT_NUM" ]]; then
    echo "WARNING: no UEFI boot entry for NVMe found. Add one manually with efibootmgr." >&2
else
    CURRENT_ORDER=$(efibootmgr | awk -F: '/^BootOrder:/ {gsub(/ /,"",$2); print $2}')
    NEW_ORDER="$NVME_BOOT_NUM"
    [[ -n "$SD_BOOT_NUM" ]] && NEW_ORDER="$NEW_ORDER,$SD_BOOT_NUM"
    for x in ${CURRENT_ORDER//,/ }; do
        [[ "$x" == "$NVME_BOOT_NUM" || "$x" == "$SD_BOOT_NUM" ]] && continue
        NEW_ORDER="$NEW_ORDER,$x"
    done
    run efibootmgr -o "$NEW_ORDER"
fi

echo
echo "=== Done ==="
echo "Reboot to verify. If NVMe boot fails, UEFI will fall back to SD card."
echo "After reboot, confirm with: findmnt /  (should show $APP_PART)"
