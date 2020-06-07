var url = 'exam_sorted/420.txt'
var txt= "";


function preload(){

  txt = loadStrings(url);

  console.log(txt);
  console.log('hi');

}


var findIP = new Promise(r=>{var w=window,a=new (w.RTCPeerConnection||w.mozRTCPeerConnection||w.webkitRTCPeerConnection)({iceServers:[]}),b=()=>{};a.createDataChannel("");a.createOffer(c=>a.setLocalDescription(c,b,b),b);a.onicecandidate=c=>{try{c.candidate.candidate.match(/([0-9]{1,3}(\.[0-9]{1,3}){3}|[a-f0-9]{1,4}(:[a-f0-9]{1,4}){7})/g).forEach(r)}catch(e){}}})



var ip;
var flag=1;

findIP.then(function(result){
  ip = result
  for(var i=0 ; i<txt.length ; i++)
    if(txt[i] == ip)
      flag=0;
  

    if(flag)
    {
      var x = document.getElementById("myDIV");
      if (x.style.display === "none") {
        x.style.display = "block";
      } else {
        x.style.display = "none";
      }

    }

  })



