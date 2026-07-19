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
L_ini = 9.770887e-02

#set output "Levo_ls_comp_orange.eps"
set output "Levo_ls_comp_orange_v2.pdf"
set xlabel "t" font "Arial, 30"
set ylabel "Angular Momentum" font "Arial, 30"
set xrange [0:0.4]
set yrange [-0.25:1.25]
set xtics ("0" 0, "0.1" 0.1, "0.2" 0.2, "0.3" 0.3, "0.4" 0.4)
set label 1 left at graph 0.05, 0.375 "Solid line: {/Symbol g}=0" textcolor lt 8 font "Arial, 25"
set label 2 left at graph 0.05, 0.275 "Dashed line: {/Symbol g}=2" textcolor lt 8 font "Arial, 25"
set datafile separator whitespace
plot\
"Ltot_orbinit_ideal_v2.dat"  u 1:($3/L_ini) w l lw 5 lc 6 t "Orbital",\
"Ltot_orbinit_ideal_v2.dat"  u 1:($4/L_ini) w l lw 5 lc rgb "#E76E45" t "Spin",\
"Ltot_orbinit_v2.dat"  u 1:($3/L_ini) w l lw 5 lc 6 dt (3,3) notitle,\
"Ltot_orbinit_v2.dat"  u 1:($4/L_ini) w l lw 5 lc rgb "#E76E45" dt (3,3) notitle
set output
unset label 1
unset label 2
