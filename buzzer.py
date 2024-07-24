import pywinusb.hid as hid
from time import sleep

USB_CFG_VENDOR_ID = 0x16C0  # Should suit, if not check ID with a tool like USBDeview
USB_CFG_DEVICE_ID = 0x05DF  # Should suit, if not check ID with a tool like USBDeview

filter = None
hid_device = None
device = None
report = None

last_row_status = None


def get_Hid_USBRelay():
    global filter, hid_device, device
    filter = hid.HidDeviceFilter(
        vendor_id=USB_CFG_VENDOR_ID, product_id=USB_CFG_DEVICE_ID
    )
    hid_device = filter.get_devices()
    device = hid_device[0]


def open_device():
    if device.is_active():
        if not device.is_opened():
            device.open()
            get_report()
            return True
        else:
            print("Device already opened")
            return True
    else:
        print("Device is not active")
    return False


def close_device():
    if device.is_active():
        if device.is_opened():
            device.close()
            return True
        else:
            print("Device already closed")
    else:
        print("Device is not active")
    return True


def refresh():
    get_Hid_USBRelay()
    open_device()


def get_report():
    global report
    if not device.is_active():
        report = None

    for rep in device.find_output_reports() + device.find_feature_reports():
        report = rep


def read_status_row():
    global last_row_status
    if report is None:
        print("Cannot read report")
        last_row_status = [0, 1, 0, 0, 0, 0, 0, 0, 3]
    else:
        last_row_status = report.get()
    return last_row_status


def write_row_data(buffer):
    if report is not None:
        report.send(raw_data=buffer)
        return True
    else:
        print("Cannot write in the report. check if your device is still plugged")
        return False


def on_all():
    if write_row_data(buffer=[0, 0xFE, 0, 0, 0, 0, 0, 0, 1]):
        return read_relay_status(relay_number=3)
    else:
        print("Cannot put ON relays")
        return False


def off_all():
    if write_row_data(buffer=[0, 0xFC, 0, 0, 0, 0, 0, 0, 1]):
        return read_relay_status(relay_number=3)
    else:
        print("Cannot put OFF relays")
        return False


def on_relay(relay_number):
    if write_row_data(buffer=[0, 0xFF, relay_number, 0, 0, 0, 0, 0, 1]):
        return read_relay_status(relay_number)
    else:
        print("Cannot put ON relay number {}".format(relay_number))
        return False


def off_relay(relay_number):
    if write_row_data(buffer=[0, 0xFD, relay_number, 0, 0, 0, 0, 0, 1]):
        return read_relay_status(relay_number)
    else:
        print("Cannot put OFF relay number {}".format(relay_number))
        return False


def read_relay_status(relay_number):
    buffer = read_status_row()
    return relay_number & buffer[8]


def is_relay_on(relay_number):
    return read_relay_status(relay_number) > 0


get_Hid_USBRelay()
open_device()


# print(" --- read_status_row: {}".format(read_status_row()))
# print("TURN OFF ALL: {}".format(off_all()))


# print("TURN ON 1: {} ".format(on_relay(1)))
# print("READ STATE 1: {}".format(read_relay_status(1)))
# sleep(3)
# print("READ STATE 1: {}".format(read_relay_status(1)))
# sleep(3)

# print("TURN ON ALL: {}".format(on_all()))
# sleep(1)
# print("TURN OFF ALL: {}".format(off_all()))
