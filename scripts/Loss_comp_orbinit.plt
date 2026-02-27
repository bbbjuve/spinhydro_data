reset
PS=1
if (PS == 0) set term x11
set terminal postscript eps enhanced color font "Arial,30"
#set terminal pdfcairo font "Arial,30" linewidth 4 enhanced size 6,4 pdfversion 1.5
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
set xlabel "Number of iterations" font "Arial, 40"
set xrange [0:20000]
set xtics ("0" 0, "0.5 {/Symbol \\264} 10^{4}" 5000, "1 {/Symbol \\264} 10^{4}" 10000, "1.5 {/Symbol \\264} 10^{4}" 15000, "2 {/Symbol \\264} 10^{4}" 20000)
set xtics font "Arial, 30"
set ytics font "Arial, 30"
##################################################################

set output "Rcl_comp_v2.eps"
#set output "Rcl_comp.pdf"
set ylabel "~R{.6-}^{C.L.}_2({/Symbol y})" font "Arial, 40"
set logscale y
set format y "10^{%L}"
plot\
"model_short_orbinit_w_penalty.dat"   u 1:4 w l lw 5 lc 7 t "w/  penalty",\
"model_short_orbinit_wo_penalty.dat"  u 1:4 w l lw 5 lc 6 t "w/o penalty"
set output  

