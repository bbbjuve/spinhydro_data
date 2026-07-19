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

#set output "integration.eps"
set output "integration_v2.pdf"
set xlabel "r" font "Arial, 30"
set ylabel "I" font "Arial, 30"
set xrange [0:1]
set yrange [-0.1:0.1]
#set xtics ("0" 0, "0.1" 0.1, "0.2" 0.2, "0.3" 0.3, "0.4" 0.4)
#set label 1 left at graph 0.05, 0.425 "Solid line: {/Symbol g}=0" textcolor lt 8 font "Arial, 25"
#set label 2 left at graph 0.05, 0.325 "Dashed line: {/Symbol g}=2" textcolor lt 8 font "Arial, 25"
f(x)=0
plot\
"Rd_avg_r_731015050.dat"  u 1:2 w l lw 5 lc 7 t "w/  penalty",\
"Rd_avg_r_731006050.dat"  u 1:2 w l lw 5 lc 6 t "w/o penalty",\
f(x) w l dt(3,3) lw 5 lc 8 notitle
set output
unset label 1
unset label 2

exit()
