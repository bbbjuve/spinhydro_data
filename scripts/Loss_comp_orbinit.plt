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
set xrange [0:100000]
set xtics ("0" 0, "2.5 {/Symbol \\264} 10^{4}" 25000, "5 {/Symbol \\264} 10^{4}" 50000, "7.5 {/Symbol \\264} 10^{4}" 75000, "10 {/Symbol \\264} 10^{4}" 100000)
set xtics font "Arial, 30"
set ytics font "Arial, 30"
##################################################################



#set output "Rcl_comp_app.eps"
set output "Rcl_comp_app.pdf"
set ylabel "~R{.6-}^{C.L.}_2({/Symbol y})" font "Arial, 40"
set logscale y
set format y "10^{%L}"
#unset xrange


## horizontal guide lines
#set arrow 101 from graph 0, first 1e-6 to graph 1, first 1e-6 nohead lc rgb "gray70" lw 1 dt 2 back
#set arrow 102 from graph 0, first 1e-5 to graph 1, first 1e-5 nohead lc rgb "gray70" lw 1 dt 2 back
#set arrow 103 from graph 0, first 1e-4 to graph 1, first 1e-4 nohead lc rgb "gray70" lw 1 dt 2 back
#set arrow 104 from graph 0, first 1e-3 to graph 1, first 1e-3 nohead lc rgb "gray70" lw 1 dt 2 back
#set arrow 105 from graph 0, first 1e-2 to graph 1, first 1e-2 nohead lc rgb "gray70" lw 1 dt 2 back

plot\
"training_log_SpinU_1.txt"        u 1:5 w l lw 5 lc 7 notitle,\
"training_log_SpinU_1_woLOSS.txt" u 1:5 w l lw 5 lc 6 notitle
set output


#unset arrow 101
#unset arrow 102
#unset arrow 103
#unset arrow 104
#unset arrow 105


exit
