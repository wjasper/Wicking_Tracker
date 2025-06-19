#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Created on Fri Jan 26 09:34:29 2024

 Copyright (c) 2025 Warren J. Jasper <wjasper@ncsu.edu>

 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 Lesser General Public License for more details.
 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

 Program to read from a Mettler Balance
 MT-SICS Interface Commands Reference Manual
 www.mt.com/xs-analytical

@author: Warren J. Jasper

Use USB C <---> USB A cable
Step 1: Settings->Services->MT-SICS service->Interface->USB
Step 2: Settings->Balance->Publishing->Transfer data->MT-SICS service is selected with "single weights"

"""
import serial.tools.list_ports
import serial
import json
import time
import datetime
import sys
import pathvalidate

from sys import exit

# Error Handling
class Error(Exception):
  ''' Base class for other exceptions.'''
  pass

class ResultError(Error):
  ''' Raised when return packet fails.'''
  pass
class mettler_MX:
    # command constands
    def __init__(self, serial_number=None):
      vendorID = 0x0eb8
      productID = 0x2110
       
      # check_for = "USB VID:PID={vid:04x}:{pid:04x}".format(vid=vendorID,pid=productID).upper()
      try:
        ports = serial.tools.list_ports.comports()
      except:
        print("error in serial.tools.list_ports.comports()")

      # iterate on devices until we find the one we want
      for check_port in ports:
        if hasattr(serial.tools,'list_ports_common'):
          print(check_port.device, hex(check_port.vid), hex(check_port.pid))
          if (check_port.vid, check_port.pid) == (vendorID, productID):
            print("Found a balance. serial_number ", serial_number, check_port.serial_number)
            try:
              self.ser = serial.Serial(check_port.device,
                                       baudrate=9600,
                                       bytesize=serial.EIGHTBITS,
                                       exclusive=True,
                                       stopbits=serial.STOPBITS_ONE,
                                       parity=serial.PARITY_NONE,
                                       xonxoff=True,
                                       rtscts=False,
                                       dsrdtr=False,
                                       timeout=2)
            except:
              continue
            if serial_number == None or serial_number == check_port.serial_number:
              return
            else:
              if self.ser.is_open:
                self.ser.close()
            continue
      raise ResultError

    def command(self, command):
      command = command + '\r\n'
      self.ser.write(command.encode("UTF-8"))
#      self.ser.flushOutput()
      time.sleep(1)
      buffer = ""
      while (self.ser.in_waiting > 0):
        buffer += self.ser.readline().decode()
#      print(buffer)  # use for debugging only
      return buffer
      
def main():
#    serial_number = "C404523175"
    serial_number = "C523137284"
    try:
        mettler = mettler_MX(serial_number)
    except:
        print("No Mettler Toledo Balance Found")
        exit(0)
        
#    print(mettler.command("I0"))
#    print(mettler.command('D "Balance"'))

    print("Serial Number: ",mettler.command("I4"))             # Query serial number
    print("Device Identification Number: ",mettler.command("I10"))  # Query device identification
    print("Software material number:", mettler.command("I5"))   # Query software material number   
#    print(mettler.command("DAT"))  # Query the date
#    print(mettler.command("TIM"))  # Query the time

    # set time
    current_time = datetime.datetime.now()
    current_time = f"TIM {current_time.hour:02d} {current_time.minute:02d} {current_time.second:02d}"
    mettler.command(current_time)
 #   print(mettler.command("TIM"))  # Query the time

    operator = input("Enter operator name: ")
    notes = input("Enter notes: ")
    current_time = datetime.datetime.now()
    date_string = current_time.strftime(" %m %d %Y %H %M %S")

    ans = input("Weigh Frame: ")
    try:
      frame_weight = float(mettler.command("S").split()[2])
    except:
      print("Weighing not stable")
      frame_weight = 0.0
    print(frame_weight)

    ans = input("Weigh Frame + Fabric: ")
    try:
      frame_plus_fabric_weight = float(mettler.command("S").split()[2])
    except:
      print("Weighing not stable")
      frame_plut_fabric_weight = 0.0
    print(frame_plus_fabric_weight)

    ans = input("Weigh Sample Wet: ")
    try:
      sample_wet_weight = float(mettler.command("SI").split()[2])
    except:
      print("Weighing not stable")
      sample_wet_weight = 0.0
    print(sample_wet_weight)

    ans = input("Weigh Sample Dry: ")
    try:
      sample_dry_weight = float(mettler.command("SI").split()[2])
    except:
      print("Weighing not stable")
      sample_dry_weight = 0.0
    print(sample_dry_weight)

    buffer = {"Date":date_string, "Operator":operator, "Notes":notes,
              "Frame_Weight":frame_weight, "Frame+Fabric_Weight":frame_plus_fabric_weight,
              "Sample_Wet_Weight":sample_wet_weight, "Sample_Dry_Weight":sample_dry_weight}

    print(buffer)
    val = input('Save data [y/n]?: ')              
    
    if val.lower() == 'y':
        filename = input('Enter filename: ')
        if not filename.endswith(".json"):
            filename += ".json"   # make sure file is .json
        if not pathvalidate.is_valid_filename(filename):
            print(f"'{filename}' not a valid filename. Sanitizing ...\n")
        with open(pathvalidate.sanitize_filename(filename),"w+") as fd:
            json.dump(buffer, fd, indent=2)

if __name__ == "__main__":
  main()
