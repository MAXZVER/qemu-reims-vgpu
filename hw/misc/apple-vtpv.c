/*
 * TEMP (lab): Apple paravirtual VideoToolbox device (PCI 106b:1a03), probe stage.
 *
 * macOS guests ship AppleVideoToolboxParavirtualization.kext, which matches
 * this PCI ID through AppleVirtIOPCITransport (a modern virtio-pci transport
 * under Apple's vendor ID, like vmapple's 106b:1a00 virtio-blk). The message
 * protocol is not public. This stage exposes the transport only and logs what
 * the guest driver does with it: negotiated features, status, config reads,
 * and every buffer it posts on every queue (sizes plus the first bytes), so
 * the protocol can be read off a boot. Buffers are held unless complete=on.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/iov.h"
#include "qemu/module.h"
#include "qapi/error.h"
#include "hw/core/qdev-properties.h"
#include "hw/pci/pci_ids.h"
#include "hw/virtio/virtio.h"
#include "hw/virtio/virtio-pci.h"
#include "standard-headers/linux/virtio_ids.h"

#define TYPE_APPLE_VTPV "apple-vtpv"
OBJECT_DECLARE_SIMPLE_TYPE(AppleVtpv, APPLE_VTPV)

#define APPLE_VTPV_MAX_QUEUES 16
#define PCI_DEVICE_ID_APPLE_VIRTIO_VTPV 0x1a03

struct AppleVtpv {
    VirtIODevice parent_obj;
    VirtQueue *vq[APPLE_VTPV_MAX_QUEUES];
    VirtQueueElement *held[APPLE_VTPV_MAX_QUEUES][64];
    uint32_t held_n[APPLE_VTPV_MAX_QUEUES];
    uint32_t queues;
    uint32_t queue_size;
    uint32_t config_size;
    uint64_t offered;
    bool complete;
    uint32_t config_reads;
};

static void vtpv_log_bytes(const char *what, const uint8_t *p, size_t n)
{
    char line[3 * 64 + 1];
    size_t i, k = 0;
    for (i = 0; i < n && i < 64; i++) {
        k += snprintf(line + k, sizeof(line) - k, "%02x ", p[i]);
    }
    line[k] = 0;
    fprintf(stderr, "apple-vtpv:   %s %zu bytes: %s\n", what, n, line);
}

static void vtpv_handle(VirtIODevice *vdev, VirtQueue *vq)
{
    AppleVtpv *s = APPLE_VTPV(vdev);
    int q = virtio_get_queue_index(vq);
    VirtQueueElement *elem;

    while ((elem = virtqueue_pop(vq, sizeof(VirtQueueElement)))) {
        size_t out = iov_size(elem->out_sg, elem->out_num);
        size_t in = iov_size(elem->in_sg, elem->in_num);
        uint8_t buf[256];
        size_t got = iov_to_buf(elem->out_sg, elem->out_num, 0, buf,
                                MIN(out, sizeof(buf)));
        fprintf(stderr, "apple-vtpv: q%d elem out_num=%u out=%zu in_num=%u in=%zu\n",
                q, elem->out_num, out, elem->in_num, in);
        for (unsigned i = 0; i < elem->out_num && i < 4; i++) {
            fprintf(stderr, "apple-vtpv:   out_sg[%u] len=%zu\n", i, elem->out_sg[i].iov_len);
        }
        for (unsigned i = 0; i < elem->in_num && i < 4; i++) {
            fprintf(stderr, "apple-vtpv:   in_sg[%u] len=%zu\n", i, elem->in_sg[i].iov_len);
        }
        if (got) {
            vtpv_log_bytes("out", buf, got);
            if (got > 64) {
                vtpv_log_bytes("out+64", buf + 64, got - 64);
            }
            if (got > 128) {
                vtpv_log_bytes("out+128", buf + 128, got - 128);
            }
        }
        if (s->complete) {
            virtqueue_push(vq, elem, 0);
            virtio_notify(vdev, vq);
            g_free(elem);
        } else if (s->held_n[q] < 64) {
            s->held[q][s->held_n[q]++] = elem;
        } else {
            virtqueue_detach_element(vq, elem, 0);
            g_free(elem);
        }
    }
}

static uint64_t vtpv_get_features(VirtIODevice *vdev, uint64_t f, Error **errp)
{
    AppleVtpv *s = APPLE_VTPV(vdev);
    fprintf(stderr, "apple-vtpv: get_features host=%#" PRIx64 " offered=%#" PRIx64 "\n",
            f, s->offered);
    return f | s->offered;
}

static void vtpv_set_features(VirtIODevice *vdev, uint64_t f)
{
    fprintf(stderr, "apple-vtpv: guest features %#" PRIx64 "\n", f);
}

static int vtpv_set_status(VirtIODevice *vdev, uint8_t status)
{
    fprintf(stderr, "apple-vtpv: status %#x\n", status);
    return 0;
}

static void vtpv_get_config(VirtIODevice *vdev, uint8_t *config)
{
    AppleVtpv *s = APPLE_VTPV(vdev);
    if (s->config_reads++ < 64) {
        fprintf(stderr, "apple-vtpv: config read #%u\n", s->config_reads);
    }
}

static void vtpv_set_config(VirtIODevice *vdev, const uint8_t *config)
{
    AppleVtpv *s = APPLE_VTPV(vdev);
    vtpv_log_bytes("config write", config, s->config_size);
}

static void vtpv_reset(VirtIODevice *vdev)
{
    AppleVtpv *s = APPLE_VTPV(vdev);
    fprintf(stderr, "apple-vtpv: reset\n");
    for (unsigned q = 0; q < s->queues; q++) {
        for (unsigned i = 0; i < s->held_n[q]; i++) {
            g_free(s->held[q][i]);
        }
        s->held_n[q] = 0;
    }
}

static void vtpv_realize(DeviceState *dev, Error **errp)
{
    VirtIODevice *vdev = VIRTIO_DEVICE(dev);
    AppleVtpv *s = APPLE_VTPV(dev);

    if (s->queues == 0 || s->queues > APPLE_VTPV_MAX_QUEUES) {
        error_setg(errp, "apple-vtpv: queues must be 1..%d", APPLE_VTPV_MAX_QUEUES);
        return;
    }
    virtio_init(vdev, VIRTIO_ID_VIDEO_DECODER, s->config_size);
    for (unsigned q = 0; q < s->queues; q++) {
        s->vq[q] = virtio_add_queue(vdev, s->queue_size, vtpv_handle);
    }
    fprintf(stderr, "apple-vtpv: realized queues=%u size=%u config=%u complete=%d\n",
            s->queues, s->queue_size, s->config_size, s->complete);
}

static void vtpv_unrealize(DeviceState *dev)
{
    VirtIODevice *vdev = VIRTIO_DEVICE(dev);
    AppleVtpv *s = APPLE_VTPV(dev);
    vtpv_reset(vdev);
    for (unsigned q = 0; q < s->queues; q++) {
        virtio_delete_queue(s->vq[q]);
    }
    virtio_cleanup(vdev);
}

static const Property vtpv_properties[] = {
    DEFINE_PROP_UINT32("queues", AppleVtpv, queues, 4),
    DEFINE_PROP_UINT32("queue-size", AppleVtpv, queue_size, 256),
    DEFINE_PROP_UINT32("config-size", AppleVtpv, config_size, 64),
    DEFINE_PROP_UINT64("features", AppleVtpv, offered, 0),
    DEFINE_PROP_BOOL("complete", AppleVtpv, complete, false),
};

static void vtpv_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

    device_class_set_props(dc, vtpv_properties);
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    vdc->realize = vtpv_realize;
    vdc->unrealize = vtpv_unrealize;
    vdc->get_features = vtpv_get_features;
    vdc->set_features = vtpv_set_features;
    vdc->set_status = vtpv_set_status;
    vdc->get_config = vtpv_get_config;
    vdc->set_config = vtpv_set_config;
    vdc->reset = vtpv_reset;
}

static const TypeInfo vtpv_info = {
    .name = TYPE_APPLE_VTPV,
    .parent = TYPE_VIRTIO_DEVICE,
    .instance_size = sizeof(AppleVtpv),
    .class_init = vtpv_class_init,
};

/* PCI proxy: a modern virtio-pci transport carrying Apple's IDs. */

#define TYPE_APPLE_VTPV_PCI "apple-vtpv-pci"
OBJECT_DECLARE_SIMPLE_TYPE(AppleVtpvPCI, APPLE_VTPV_PCI)

struct AppleVtpvPCI {
    VirtIOPCIProxy parent_obj;
    AppleVtpv vdev;
};

static void vtpv_pci_realize(VirtIOPCIProxy *vpci_dev, Error **errp)
{
    AppleVtpvPCI *dev = APPLE_VTPV_PCI(vpci_dev);
    DeviceState *vdev = DEVICE(&dev->vdev);

    if (vpci_dev->nvectors == DEV_NVECTORS_UNSPECIFIED) {
        vpci_dev->nvectors = dev->vdev.queues + 1;
    }
    virtio_pci_force_virtio_1(vpci_dev);
    if (!qdev_realize(vdev, BUS(&vpci_dev->bus), errp)) {
        return;
    }
    pci_config_set_vendor_id(vpci_dev->pci_dev.config, PCI_VENDOR_ID_APPLE);
    pci_config_set_device_id(vpci_dev->pci_dev.config, PCI_DEVICE_ID_APPLE_VIRTIO_VTPV);
}

static const Property vtpv_pci_properties[] = {
    DEFINE_PROP_UINT32("vectors", VirtIOPCIProxy, nvectors, DEV_NVECTORS_UNSPECIFIED),
};

static void vtpv_pci_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtioPCIClass *k = VIRTIO_PCI_CLASS(klass);
    PCIDeviceClass *pcidev_k = PCI_DEVICE_CLASS(klass);

    device_class_set_props(dc, vtpv_pci_properties);
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    k->realize = vtpv_pci_realize;
    pcidev_k->vendor_id = PCI_VENDOR_ID_APPLE;
    pcidev_k->device_id = PCI_DEVICE_ID_APPLE_VIRTIO_VTPV;
    pcidev_k->revision = VIRTIO_PCI_ABI_VERSION;
    pcidev_k->class_id = PCI_CLASS_MULTIMEDIA_VIDEO;
}

static void vtpv_pci_instance_init(Object *obj)
{
    AppleVtpvPCI *dev = APPLE_VTPV_PCI(obj);
    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev), TYPE_APPLE_VTPV);
}

static const VirtioPCIDeviceTypeInfo vtpv_pci_info = {
    .generic_name = TYPE_APPLE_VTPV_PCI,
    .instance_size = sizeof(AppleVtpvPCI),
    .instance_init = vtpv_pci_instance_init,
    .class_init = vtpv_pci_class_init,
};

static void vtpv_register_types(void)
{
    type_register_static(&vtpv_info);
    virtio_pci_types_register(&vtpv_pci_info);
}

type_init(vtpv_register_types)
