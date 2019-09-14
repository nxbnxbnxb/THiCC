#!/bin/bash

#===============================================================================
# TODO: TODO: TODO:  TODO: TODO: TODO:  TODO: TODO: TODO:  TODO: TODO: TODO: 
#   nxb tested this on his own Thinkpad t420 laptop.  
#   He makes NO GUARANTEES that this will work on any other version of blender. 
# NOTE: NOTE: NOTE: NOTE: NOTE: NOTE:  NOTE: NOTE: NOTE:  NOTE: NOTE: NOTE: 
#===============================================================================

# xdotool is unreliable.  -nxb, Mon Sep  9 04:06:49 EDT 2019


# It worked for some reason after I got rid of the "open blender" command.  So I guess I'll just forget about "why?" and finish the job.
#./blender.sh
#===================================================================================================
#sleep 3.0
sleep 0.05

#===================================================================================================
# Click "File => Import => .obj (Wavefront)" from the menu:
#===================================================================================================
xdotool mousemove 50 30 # x, y      ((0,0) is the top left corner)
sleep 0.05
xdotool mousedown 1 # '1' is the     main (left) mouse "click" button
sleep 0.05
xdotool mouseup 1
sleep 0.05
xdotool key i # "import" in Blender menu.
sleep 0.05
xdotool key w # ".obj (Wavefront)" in Blender menu.
sleep 0.05

#===================================================================================================
# Find the .obj file from the local .obj mesh bucket:
#===================================================================================================
# Move mouse over the "filepath-picker" (the GUI that lets the user type in a filename so blender can pen the .obj file.)
xdotool mousemove 300 90
sleep 0.05
# Double-click:
xdotool mousedown 1 # '1' is the     main (left) mouse "click" button
sleep 0.05
xdotool mouseup   1
sleep 0.05
#xdotool mousedown 1 # '1' is the     main (left) mouse "click" button
#sleep 0.05
#xdotool mouseup   1
#sleep 0.05
#===================================================================================================
# Type "/home/n/V/000.obj" :
#===================================================================================================
#   NOTE: 
#     Design choice: I decided not to use the mouse b/c blender doesn't always starts the user in the same directory.
#===================================================================================================
sleep 3.0
xdotool type '/home/n/V/000.obj'
sleep 1.0
#xdotool key slash h o m e slash n slash V slash 0 0 0 period o b j Return

#sleep 0.35
#xdotool key slash
#sleep 0.35
#xdotool key h
#sleep 0.35
#xdotool key o
#sleep 0.35
#xdotool key m
#sleep 0.35
#xdotool key e
#sleep 0.35
#xdotool key slash
#sleep 0.35
#xdotool key n
#sleep 0.35
#xdotool key slash
#sleep 0.35
#xdotool key V
#sleep 0.35
#xdotool key slash
#sleep 0.35
#xdotool key 0
#sleep 0.35
#xdotool key 0
#sleep 0.35
#xdotool key 0
#sleep 0.35
#xdotool key period
#sleep 0.35
#xdotool key o
#sleep 0.35
#xdotool key b
#sleep 0.35
#xdotool key j
#sleep 0.35
xdotool key Return
sleep 0.35
xdotool key Return # NOTE:  We need "double-enter" to import the .obj mesh.

#===================================================================================================
# Change view to "top" view:
#===================================================================================================
xdotool mousemove 60 680
xdotool mousedown 1
xdotool mouseup 1
xdotool key o # "top"
#===================================================================================================
# Rotate SMPL-X mesh s.t. it faces the camera:
#===================================================================================================
# move over 'Rotate' 
xdotool mousemove 60 110
sleep 0.1
# click 'Rotate' 
xdotool mousedown 1
sleep 0.1
xdotool mouseup 1
sleep 0.1
# Rotate towards the camera in the lower left:
for i in {111..670..1}
do
  xdotool mousemove 60 $i
  sleep 0.01
done
xdotool mousedown 1
xdotool mouseup 1





















