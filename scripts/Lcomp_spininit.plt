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
#set datafile separator ","
L_ini = 2.356195e-01

#set output "Levo_sl_comp_orange.eps"
set output "Levo_sl_comp_orange_v2.pdf"
set xlabel "t" font "Arial, 30"
set ylabel "Angular Momentum" font "Arial, 30"
set xrange [0:0.4]
set yrange [-0.25:1.25]
set xtics ("0" 0, "0.1" 0.1, "0.2" 0.2, "0.3" 0.3, "0.4" 0.4)
set label 1 left at graph 0.05, 0.425 "Solid line: {/Symbol g}=0" textcolor lt 8 font "Arial, 25"
set label 2 left at graph 0.05, 0.325 "Dashed line: {/Symbol g}=2" textcolor lt 8 font "Arial, 25"
set datafile separator whitespace
l(x)=1
s(x)=0
plot \
s(x) w l lw 5 lc 6 t "Orbital",\
l(x) w l lw 5 lc rgb "#E76E45" t "Spin",\
"Ltot_spininit_v2.dat"  u 1:($3/L_ini) w l lw 5 lc 6 dt (3,3) notitle,\
"Ltot_spininit_v2.dat"  u 1:($4/L_ini) w l lw 5 lc rgb "#E76E45" dt (3,3) notitle
set output
unset label 1
unset label 2
