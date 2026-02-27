reset
PS=1
if (PS == 0) set term x11
set terminal postscript eps enhanced color font "Arial,30"
##################################################################
set xtics nomirror
set ytics nomirror
set xtics auto
set ytics auto
unset logscale x
unset logscale y
set key above
unset xrange
unset yrange
unset xlabel
unset ylabel
unset format y
set key  font "Arial, 20"
set tics font "Arial, 20"
##################################################################
#set label 1 right at graph 0.95, 0.9 "g=2" textcolor lt 8 font "Arial, 25"
#set format y "10^{%L}"
#set format y "%.0e"
#set datafile separator ","
##################################################################

set output "spectrum.eps"
#set output "spectrum.pdf"
set xlabel "f" font "Arial, 30"
set ylabel "{/Symbol r}" font "Arial, 30"
set xrange [0:5]
#set yrange [-0.25:1.25]
#set xtics ("0" 0, "0.1" 0.1, "0.2" 0.2, "0.3" 0.3, "0.4" 0.4)
#set label 1 left at graph 0.05, 0.425 "Solid line: {/Symbol g}=0" textcolor lt 8 font "Arial, 25"
#set label 2 left at graph 0.05, 0.325 "Dashed line: {/Symbol g}=2" textcolor lt 8 font "Arial, 25"
plot\
"Rd_spectrum_731015050.dat"  u 1:2 w l lw 5 lc 7 t "w/  penalty",\
"Rd_spectrum_731006050.dat"  u 1:2 w l lw 5 lc 6 t "w/o penalty"
set output  
unset label 1
unset label 2

exit()
