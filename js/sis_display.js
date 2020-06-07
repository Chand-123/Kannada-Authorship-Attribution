var txt;
var str1;
var name;
var usn,pass;
var data;
var n1,n2,n3;
var string = window.location.search;


var res = string.split("&");

usn = res[0].replace("?usn=","");
pass = res[1].replace("pass=","")
