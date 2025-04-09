#!/bin/bash
# system_setup.sh
# Configuration script for setting up the deterministic derivatives pricing system
#
# This script performs the following:
# 1. Isolates CPU cores for the pricing system
# 2. Configures the Linux kernel for real-time performance
# 3. Allocates huge pages
# 4. Sets up the necessary permissions for performance counters
# 5. Disables CPU frequency scaling
#
# Usage: sudo ./system_setup.sh

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root. Please use: sudo $0"
    exit 1
fi

echo "==================================================="
echo "Deterministic Derivatives Pricing System Setup"
echo "==================================================="

# ------------------------------------
# 1. CPU Isolation Setup
# ------------------------------------
echo "Setting up CPU isolation..."

# Backup the existing grub configuration
cp /etc/default/grub /etc/default/grub.bak
echo "Original grub config backed up to /etc/default/grub.bak"

# Add CPU isolation parameters to grub
# Isolate cores 1-4 for our application
sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 isolcpus=1-4 nohz_full=1-4 rcu_nocbs=1-4"/' /etc/default/grub

# Update grub
update-grub

# ------------------------------------
# 2. IRQ Balancing
# ------------------------------------
echo "Disabling IRQ balancing for isolated cores..."

# Stop the irqbalance service
systemctl stop irqbalance
systemctl disable irqbalance

# Create config to exclude our isolated cores
cat > /etc/default/irqbalance << EOF
# Configuration for the irqbalance daemon
ENABLED="1"
OPTIONS="--banirq=timer --banirq=cpuidle --ban_outbound_affinities=1-4"
EOF

# ------------------------------------
# 3. CPU Frequency Scaling
# ------------------------------------
echo "Setting CPU governor to performance..."

# Set governor to performance for all CPUs
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $governor
done

# Disable Intel P-states for more consistent performance
if [ -f /sys/devices/system/cpu/intel_pstate/status ]; then
    echo passive > /sys/devices/system/cpu/intel_pstate/status
fi

# ------------------------------------
# 4. Huge Pages Setup
# ------------------------------------
echo "Configuring huge pages..."

# Create mount point for huge pages
mkdir -p /mnt/huge

# Configure 1024 huge pages at 2MB each (2GB total)
echo 1024 > /proc/sys/vm/nr_hugepages

# Add to fstab for persistence
if ! grep -q "hugetlbfs" /etc/fstab; then
    echo "hugetlbfs /mnt/huge hugetlbfs defaults 0 0" >> /etc/fstab
fi

# Mount huge pages
mount -t hugetlbfs nodev /mnt/huge

# ------------------------------------
# 5. System Limits Configuration
# ------------------------------------
echo "Configuring system limits..."

# Add limits configuration
cat > /etc/security/limits.d/99-pricing-system.conf << EOF
# Limits for deterministic pricing system
*               soft    memlock         unlimited
*               hard    memlock         unlimited
*               soft    rtprio          99
*               hard    rtprio          99
*               soft    nice            -20
*               hard    nice            -20
EOF

# ------------------------------------
# 6. Network Configuration
# ------------------------------------
echo "Configuring network for low latency..."

# Set network tuning parameters
cat > /etc/sysctl.d/99-network-tuning.conf << EOF
# Network tuning for low latency
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 16777216
net.core.wmem_default = 16777216
net.core.netdev_max_backlog = 30000
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 87380 16777216
net.ipv4.tcp_tw_reuse = 1
EOF

# Apply sysctl settings
sysctl -p /etc/sysctl.d/99-network-tuning.conf

# ------------------------------------
# 7. Performance Monitoring Setup
# ------------------------------------
echo "Setting up performance monitoring permissions..."

# Allow access to perf events
echo -1 > /proc/sys/kernel/perf_event_paranoid

# Create a sysctl config file for perf settings
cat > /etc/sysctl.d/99-perf.conf << EOF
# Performance monitoring settings
kernel.perf_event_paranoid = -1
kernel.kptr_restrict = 0
EOF

# Apply sysctl settings
sysctl -p /etc/sysctl.d/99-perf.conf

# ------------------------------------
# 8. Compile and Install the System
# ------------------------------------
echo "Setting up build environment..."

# Install dependencies - assumes a Debian/Ubuntu system
apt-get update
apt-get install -y build-essential cmake libgsl-dev libeigen3-dev

# Add to xmake.lua for our project
cat > xmake.lua.addition << EOF
-- Add real-time library
add_links("rt", "pthread")

-- Enable deterministic mode
add_defines("DETERMINISTIC_MODE")

-- Set optimization flags for deterministic behavior
if is_mode("release") then
    add_cxflags("-O3", "-march=native", "-fno-strict-aliasing")
    add_cxflags("-fno-omit-frame-pointer", "-fno-math-errno")
    add_cxflags("-fno-exceptions", "-fno-rtti")
end
EOF

echo "Configuration script completed."
echo "Please reboot the system for CPU isolation settings to take effect."
echo "After reboot, compile the system with 'xmake' and run with 'sudo ./pricing_system'"