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

unset key
set output "loss_v3.eps"
#set output "loss.pdf"
set ylabel "L({/Symbol y},{/Symbol s})" font "Arial, 40"
set yrange[-6:1]
plot\
"Resi_orbinit.dat" u 1:2 w l lw 5 lc 7 notitle
set output  
unset yrange

set key above
set key  font "Arial, 30"
set output "Rge_v3.eps"
#set output "Rge.pdf"
set ylabel "~R{.6-}^{G.E.}_{sum}({/Symbol y})" font "Arial, 40"
set logscale y
set format y "10^{%L}"
plot\
"Resi_orbinit.dat" u 1:(0.2*($5+$6+$7+$8+$9)) w l lw 5 lc 7 notitle
set output  
unset key

unset format y
unset logscale y
set datafile separator ","
L_ini = 9.770887e-02
set output "Levo_v3.eps"
#set output "Levo.pdf"
set xlabel "t" font "Arial, 30"
set ylabel "Angular Momentum" font "Arial, 30"
set datafile separator whitespace
set xrange [0:0.4]
set xtics ("0" 0, "0.1" 0.1, "0.2" 0.2, "0.3" 0.3, "0.4" 0.4)
plot\
"Levo_orbinit.dat"  u 1:($2/L_ini) w l lw 7 lc 7 notitle
set output  
unset label 1

exit

