/*
 *  Apple SMC controller
 *
 *  Copyright (c) 2007 Alexander Graf
 *
 *  Authors: Alexander Graf <agraf@suse.de>
 *           Susanne Graf <suse@csgraf.de>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * *****************************************************************
 *
 * In all Intel-based Apple hardware there is an SMC chip to control the
 * backlight, fans and several other generic device parameters. It also
 * contains the magic keys used to dongle Mac OS X to the device.
 *
 * This driver was mostly created by looking at the Linux AppleSMC driver
 * implementation and does not support IRQ.
 *
 */

#include "qemu/osdep.h"
#include "hw/isa/isa.h"
#include "hw/core/qdev-properties.h"
#include "ui/console.h"
#include "qemu/error-report.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qemu/bswap.h"
#include "qom/object.h"
#include "hw/acpi/acpi_aml_interface.h"

/* #define DEBUG_SMC */

#define APPLESMC_DEFAULT_IOBASE        0x300
#define TYPE_APPLE_SMC "isa-applesmc"
#define APPLESMC_MAX_DATA_LENGTH       32
#define APPLESMC_PROP_IO_BASE "iobase"

enum {
    APPLESMC_DATA_PORT               = 0x00,
    APPLESMC_CMD_PORT                = 0x04,
    APPLESMC_ERR_PORT                = 0x1e,
    APPLESMC_NUM_PORTS               = 0x20,
};

enum {
    APPLESMC_READ_CMD                = 0x10,
    APPLESMC_WRITE_CMD               = 0x11,
    APPLESMC_GET_KEY_BY_INDEX_CMD    = 0x12,
    APPLESMC_GET_KEY_TYPE_CMD        = 0x13,
};

enum {
    APPLESMC_ST_CMD_DONE             = 0x00,
    APPLESMC_ST_DATA_READY           = 0x01,
    APPLESMC_ST_BUSY                 = 0x02,
    APPLESMC_ST_ACK                  = 0x04,
    APPLESMC_ST_NEW_CMD              = 0x08,
};

enum {
    APPLESMC_ST_1E_CMD_INTRUPTED     = 0x80,
    APPLESMC_ST_1E_STILL_BAD_CMD     = 0x81,
    APPLESMC_ST_1E_BAD_CMD           = 0x82,
    APPLESMC_ST_1E_NOEXIST           = 0x84,
    APPLESMC_ST_1E_WRITEONLY         = 0x85,
    APPLESMC_ST_1E_READONLY          = 0x86,
    APPLESMC_ST_1E_BAD_INDEX         = 0xb8,
};

#ifdef DEBUG_SMC
#define smc_debug(...) fprintf(stderr, "AppleSMC: " __VA_ARGS__)
#else
#define smc_debug(...) do { } while (0)
#endif

static char default_osk[64] = "This is a dummy key. Enter the real key "
                              "using the -osk parameter";

struct AppleSMCData {
    uint8_t len;
    const char *key;
    const char *type;
    const char *data;
    QLIST_ENTRY(AppleSMCData) node;
};

#define APPLESMC_MAX_KEYS 16

OBJECT_DECLARE_SIMPLE_TYPE(AppleSMCState, APPLE_SMC)

struct AppleSMCState {
    ISADevice parent_obj;

    MemoryRegion io_data;
    MemoryRegion io_cmd;
    MemoryRegion io_err;
    uint32_t iobase;
    uint8_t cmd;
    uint8_t status;
    uint8_t status_1e;
    uint8_t last_ret;
    char key[4];
    uint8_t read_pos;
    uint8_t data_len;
    uint8_t data_pos;
    uint8_t data[255];
    char *osk;
    QLIST_HEAD(, AppleSMCData) data_def;
    /* The keys in the SMC's own order (ascending), for GET_KEY_BY_INDEX. */
    const struct AppleSMCData *sorted[APPLESMC_MAX_KEYS];
    uint32_t nkeys;
    uint8_t nkeys_be[4];
};

/*
 * TEMP lab (devices agent, 2026-10-07): QEMU_LAB_SMC=1 prints, once a second,
 * the SMC commands the guest sent and the keys it read. The idle Tahoe guest
 * drives ~200k I/O-port exits a second on this device.
 */
static void applesmc_lab_count(uint8_t cmd, const uint8_t *key)
{
    static int on = -1;
    static int64_t last;
    static uint32_t cmds[256];
    static char keys[8][5];
    static uint32_t keyn[8];
    int64_t now;
    int i;

    if (on < 0) {
        const char *e = getenv("QEMU_LAB_SMC");
        on = e && *e == '1';
    }
    if (!on) {
        return;
    }
    if (key) {
        for (i = 0; i < 8; i++) {
            if (!keyn[i] || !memcmp(keys[i], key, 4)) {
                memcpy(keys[i], key, 4);
                keys[i][4] = 0;
                keyn[i]++;
                break;
            }
        }
    } else {
        cmds[cmd]++;
    }
    now = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
    if (now - last >= 1000) {
        GString *g = g_string_new("lab-smc cmds");
        for (i = 0; i < 256; i++) {
            if (cmds[i]) {
                g_string_append_printf(g, " 0x%02x:%u", i, cmds[i]);
                cmds[i] = 0;
            }
        }
        g_string_append(g, " keys");
        for (i = 0; i < 8; i++) {
            if (keyn[i]) {
                g_string_append_printf(g, " '%s':%u", keys[i], keyn[i]);
                keyn[i] = 0;
            }
        }
        fprintf(stderr, "%s\n", g->str);
        g_string_free(g, true);
        last = now;
    }
}

static void applesmc_io_cmd_write(void *opaque, hwaddr addr, uint64_t val,
                                  unsigned size)
{
    AppleSMCState *s = opaque;
    uint8_t status = s->status & 0x0f;

    smc_debug("CMD received: 0x%02x\n", (uint8_t)val);
    applesmc_lab_count((uint8_t)val, NULL);
    if (getenv("QEMU_LAB_SMC_OLD") && val != APPLESMC_READ_CMD) {
        val = 0xff;     /* TEMP lab A/B: the old device, which refused these */
    }
    switch (val) {
    case APPLESMC_READ_CMD:
    case APPLESMC_GET_KEY_BY_INDEX_CMD:
    case APPLESMC_GET_KEY_TYPE_CMD:
        /* did last command run through OK? */
        if (status == APPLESMC_ST_CMD_DONE || status == APPLESMC_ST_NEW_CMD) {
            s->cmd = val;
            s->status = APPLESMC_ST_NEW_CMD | APPLESMC_ST_ACK;
        } else {
            smc_debug("ERROR: previous command interrupted!\n");
            s->status = APPLESMC_ST_NEW_CMD;
            s->status_1e = APPLESMC_ST_1E_CMD_INTRUPTED;
        }
        break;
    default:
        smc_debug("UNEXPECTED CMD 0x%02x\n", (uint8_t)val);
        s->status = APPLESMC_ST_NEW_CMD;
        s->status_1e = APPLESMC_ST_1E_BAD_CMD;
    }
    s->read_pos = 0;
    s->data_pos = 0;
}

static const struct AppleSMCData *applesmc_find_key(AppleSMCState *s)
{
    struct AppleSMCData *d;

    QLIST_FOREACH(d, &s->data_def, node) {
        if (!memcmp(d->key, s->key, 4)) {
            return d;
        }
    }
    return NULL;
}

static void applesmc_io_data_write(void *opaque, hwaddr addr, uint64_t val,
                                   unsigned size)
{
    AppleSMCState *s = opaque;
    const struct AppleSMCData *d;

    smc_debug("DATA received: 0x%02x\n", (uint8_t)val);
    switch (s->cmd) {
    case APPLESMC_GET_KEY_BY_INDEX_CMD:
    case APPLESMC_GET_KEY_TYPE_CMD:
        /*
         * Four bytes (the big-endian index, or the key), then the answer.
         * macOS sends no length byte here: it waits for the answer after the
         * fourth byte (with one expected, the idle Tahoe guest polled the
         * status ~8000 times a second until its timeout). A length byte that
         * does come is taken and ignored. A guest that cannot enumerate the
         * keys retries without end: before this command was answered at all
         * the idle guest sent GET_KEY_BY_INDEX ~34000 times a second, ~200k
         * port exits.
         */
        if ((s->status & 0x0f) == APPLESMC_ST_CMD_DONE) {
            break;
        }
        if (s->read_pos < 3) {
            s->key[s->read_pos] = val;
            s->status = APPLESMC_ST_ACK;
        } else if (s->read_pos == 3) {
            s->key[3] = val;
            if (s->cmd == APPLESMC_GET_KEY_BY_INDEX_CMD) {
                uint32_t i = ldl_be_p(s->key);

                d = i < s->nkeys ? s->sorted[i] : NULL;
                if (d) {
                    memcpy(s->data, d->key, 4);
                    s->data_len = 4;
                }
            } else {
                d = applesmc_find_key(s);
                if (d) {
                    s->data[0] = d->len;
                    memcpy(s->data + 1, d->type, 4);
                    s->data[5] = 0x80;          /* readable */
                    s->data_len = 6;
                }
            }
            s->data_pos = 0;
            if (d) {
                s->status = APPLESMC_ST_ACK | APPLESMC_ST_DATA_READY;
                s->status_1e = APPLESMC_ST_CMD_DONE;
            } else {
                s->status = APPLESMC_ST_CMD_DONE;
                s->status_1e = s->cmd == APPLESMC_GET_KEY_BY_INDEX_CMD ?
                    APPLESMC_ST_1E_BAD_INDEX : APPLESMC_ST_1E_NOEXIST;
            }
        }
        s->read_pos++;
        break;
    case APPLESMC_READ_CMD:
        if ((s->status & 0x0f) == APPLESMC_ST_CMD_DONE) {
            break;
        }
        if (s->read_pos < 4) {
            s->key[s->read_pos] = val;
            s->status = APPLESMC_ST_ACK;
        } else if (s->read_pos == 4) {
            applesmc_lab_count(0, (const uint8_t *)s->key);
            d = applesmc_find_key(s);
            if (d != NULL) {
                memcpy(s->data, d->data, d->len);
                s->data_len = d->len;
                s->data_pos = 0;
                s->status = APPLESMC_ST_ACK | APPLESMC_ST_DATA_READY;
                s->status_1e = APPLESMC_ST_CMD_DONE;  /* clear on valid key */
            } else {
                smc_debug("READ_CMD: key '%c%c%c%c' not found!\n",
                          s->key[0], s->key[1], s->key[2], s->key[3]);
                s->status = APPLESMC_ST_CMD_DONE;
                s->status_1e = APPLESMC_ST_1E_NOEXIST;
            }
        }
        s->read_pos++;
        break;
    default:
        s->status = APPLESMC_ST_CMD_DONE;
        s->status_1e = APPLESMC_ST_1E_STILL_BAD_CMD;
    }
}

static void applesmc_io_err_write(void *opaque, hwaddr addr, uint64_t val,
                                  unsigned size)
{
    smc_debug("ERR_CODE received: 0x%02x, ignoring!\n", (uint8_t)val);
    /* NOTE: writing to the error port not supported! */
}

static uint64_t applesmc_io_data_read(void *opaque, hwaddr addr, unsigned size)
{
    AppleSMCState *s = opaque;

    switch (s->cmd) {
    case APPLESMC_READ_CMD:
    case APPLESMC_GET_KEY_BY_INDEX_CMD:
    case APPLESMC_GET_KEY_TYPE_CMD:
        if (!(s->status & APPLESMC_ST_DATA_READY)) {
            break;
        }
        if (s->data_pos < s->data_len) {
            s->last_ret = s->data[s->data_pos];
            smc_debug("READ '%c%c%c%c'[%d] = %02x\n",
                      s->key[0], s->key[1], s->key[2], s->key[3],
                      s->data_pos, s->last_ret);
            s->data_pos++;
            if (s->data_pos == s->data_len) {
                s->status = APPLESMC_ST_CMD_DONE;
                smc_debug("READ '%c%c%c%c' Len=%d complete!\n",
                          s->key[0], s->key[1], s->key[2], s->key[3],
                          s->data_len);
            } else {
                s->status = APPLESMC_ST_ACK | APPLESMC_ST_DATA_READY;
            }
        }
        break;
    default:
        s->status = APPLESMC_ST_CMD_DONE;
        s->status_1e = APPLESMC_ST_1E_STILL_BAD_CMD;
    }
    smc_debug("DATA sent: 0x%02x\n", s->last_ret);

    return s->last_ret;
}

static uint64_t applesmc_io_cmd_read(void *opaque, hwaddr addr, unsigned size)
{
    AppleSMCState *s = opaque;

    smc_debug("CMD sent: 0x%02x\n", s->status);
    if (getenv("QEMU_LAB_SMC")) {
        /* TEMP lab: status polls by (command, status, key, step) */
        static uint32_t n;
        static char last[64];
        static int64_t t;
        char now[64];
        int64_t ms = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

        snprintf(now, sizeof(now), "cmd=0x%02x st=0x%02x e=0x%02x key='%.4s' pos=%u/%u",
                 s->cmd, s->status, s->status_1e, s->key, s->read_pos,
                 s->data_pos);
        if (strcmp(now, last) || ms - t >= 1000) {
            if (n > 50) {
                fprintf(stderr, "lab-smc polls %u x %s\n", n, last);
            }
            n = 0;
            t = ms;
            memcpy(last, now, sizeof(last));
        }
        n++;
    }
    return s->status;
}

static uint64_t applesmc_io_err_read(void *opaque, hwaddr addr, unsigned size)
{
    AppleSMCState *s = opaque;

    /* NOTE: read does not clear the 1e status */
    smc_debug("ERR_CODE sent: 0x%02x\n", s->status_1e);
    return s->status_1e;
}

static void applesmc_add_key(AppleSMCState *s, const char *key,
                             const char *type, int len, const char *data)
{
    struct AppleSMCData *def;

    def = g_new0(struct AppleSMCData, 1);
    def->key = key;
    def->type = type;
    def->len = len;
    def->data = data;

    QLIST_INSERT_HEAD(&s->data_def, def, node);
}

static int applesmc_key_cmp(const void *a, const void *b)
{
    const struct AppleSMCData *x = *(const struct AppleSMCData *const *)a;
    const struct AppleSMCData *y = *(const struct AppleSMCData *const *)b;

    return memcmp(x->key, y->key, 4);
}

/* "#KEY" (the key count) and the ascending index GET_KEY_BY_INDEX walks. */
static void applesmc_index_keys(AppleSMCState *s)
{
    struct AppleSMCData *d;

    s->nkeys = 1;                       /* "#KEY" itself */
    QLIST_FOREACH(d, &s->data_def, node) {
        s->nkeys++;
    }
    assert(s->nkeys <= APPLESMC_MAX_KEYS);
    stl_be_p(s->nkeys_be, s->nkeys);
    applesmc_add_key(s, "#KEY", "ui32", 4, (const char *)s->nkeys_be);
    s->nkeys = 0;
    QLIST_FOREACH(d, &s->data_def, node) {
        s->sorted[s->nkeys++] = d;
    }
    qsort(s->sorted, s->nkeys, sizeof(s->sorted[0]), applesmc_key_cmp);
}

static void qdev_applesmc_isa_reset(DeviceState *dev)
{
    AppleSMCState *s = APPLE_SMC(dev);

    s->status = 0x00;
    s->status_1e = 0x00;
    s->last_ret = 0x00;
}

static const MemoryRegionOps applesmc_data_io_ops = {
    .write = applesmc_io_data_write,
    .read = applesmc_io_data_read,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 1,
    },
};

static const MemoryRegionOps applesmc_cmd_io_ops = {
    .write = applesmc_io_cmd_write,
    .read = applesmc_io_cmd_read,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 1,
    },
};

static const MemoryRegionOps applesmc_err_io_ops = {
    .write = applesmc_io_err_write,
    .read = applesmc_io_err_read,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 1,
    },
};

static void applesmc_isa_realize(DeviceState *dev, Error **errp)
{
    AppleSMCState *s = APPLE_SMC(dev);

    memory_region_init_io(&s->io_data, OBJECT(s), &applesmc_data_io_ops, s,
                          "applesmc-data", 1);
    isa_register_ioport(&s->parent_obj, &s->io_data,
                        s->iobase + APPLESMC_DATA_PORT);

    memory_region_init_io(&s->io_cmd, OBJECT(s), &applesmc_cmd_io_ops, s,
                          "applesmc-cmd", 1);
    isa_register_ioport(&s->parent_obj, &s->io_cmd,
                        s->iobase + APPLESMC_CMD_PORT);

    memory_region_init_io(&s->io_err, OBJECT(s), &applesmc_err_io_ops, s,
                          "applesmc-err", 1);
    isa_register_ioport(&s->parent_obj, &s->io_err,
                        s->iobase + APPLESMC_ERR_PORT);

    if (!s->osk || (strlen(s->osk) != 64)) {
        warn_report("Using AppleSMC with invalid key");
        s->osk = default_osk;
    }

    QLIST_INIT(&s->data_def);
    applesmc_add_key(s, "REV ", "{rev", 6, "\x01\x13\x0f\x00\x00\x03");
    applesmc_add_key(s, "OSK0", "ch8*", 32, s->osk);
    applesmc_add_key(s, "OSK1", "ch8*", 32, s->osk + 32);
    applesmc_add_key(s, "NATJ", "ui8 ", 1, "\0");
    applesmc_add_key(s, "MSSP", "ui8 ", 1, "\0");
    applesmc_add_key(s, "MSSD", "si8 ", 1, "\0x3");
    applesmc_index_keys(s);
}

static void applesmc_unrealize(DeviceState *dev)
{
    AppleSMCState *s = APPLE_SMC(dev);
    struct AppleSMCData *d, *next;

    /* Remove existing entries */
    QLIST_FOREACH_SAFE(d, &s->data_def, node, next) {
        QLIST_REMOVE(d, node);
        g_free(d);
    }
}

static const Property applesmc_isa_properties[] = {
    DEFINE_PROP_UINT32(APPLESMC_PROP_IO_BASE, AppleSMCState, iobase,
                       APPLESMC_DEFAULT_IOBASE),
    DEFINE_PROP_STRING("osk", AppleSMCState, osk),
};

static void build_applesmc_aml(AcpiDevAmlIf *adev, Aml *scope)
{
    Aml *crs;
    AppleSMCState *s = APPLE_SMC(adev);
    uint32_t iobase = s->iobase;
    Aml *dev = aml_device("SMC");

    aml_append(dev, aml_name_decl("_HID", aml_eisaid("APP0001")));
    /* device present, functioning, decoding, not shown in UI */
    aml_append(dev, aml_name_decl("_STA", aml_int(0xB)));
    crs = aml_resource_template();
    aml_append(crs,
        aml_io(AML_DECODE16, iobase, iobase, 0x01, APPLESMC_MAX_DATA_LENGTH)
    );
    aml_append(crs, aml_irq_no_flags(6));
    aml_append(dev, aml_name_decl("_CRS", crs));
    aml_append(scope, dev);
}

static void qdev_applesmc_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    AcpiDevAmlIfClass *adevc = ACPI_DEV_AML_IF_CLASS(klass);

    dc->realize = applesmc_isa_realize;
    dc->unrealize = applesmc_unrealize;
    device_class_set_legacy_reset(dc, qdev_applesmc_isa_reset);
    device_class_set_props(dc, applesmc_isa_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
    adevc->build_dev_aml = build_applesmc_aml;
}

static const TypeInfo applesmc_isa_info = {
    .name          = TYPE_APPLE_SMC,
    .parent        = TYPE_ISA_DEVICE,
    .instance_size = sizeof(AppleSMCState),
    .class_init    = qdev_applesmc_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { TYPE_ACPI_DEV_AML_IF },
        { },
    },
};

static void applesmc_register_types(void)
{
    type_register_static(&applesmc_isa_info);
}

type_init(applesmc_register_types)
