var txt;
var str1;
var name;
var data;
var n1,n2,n3;
var string = window.location.search;
// var url = "http://localhost:8000/web_html/examdata.txt";
var url = "examdata.txt";
var num;

// function preload(){
// 	txt = loadStrings(url);



// }

function setup(){
	noCanvas();
	
	str1 = string.match('usn=')
	if(str1 == 'usn='){
		byusn();
	}
	else{

		str1 = string.match('branch=')
		if(str1 == 'branch='){

			bybranch();
		}
		else
		{
			byname();
		}

	}


}




function byname(){
	loadFile();
	
}


function loadFile(){
	loadStrings(url,fileLoaded);
}

function fileLoaded(txt){

	var name = window.localStorage.getItem("id");


	
	for(var i=0; i<txt.length;i++)
		if (name == txt[i]) {
			console.log(txt);
			str1 ="";

			var node = document.createElement("th");
			var textnode = document.createTextNode("Subject");
			node.appendChild(textnode);
			document.getElementById("header").appendChild(node);
			var node = document.createElement("th");
			var textnode = document.createTextNode("Subject Code");
			node.appendChild(textnode);
			document.getElementById("header").appendChild(node);
			var node = document.createElement("th");
			var textnode = document.createTextNode("Crdeits Required");
			node.appendChild(textnode);
			document.getElementById("header").appendChild(node);
			var node = document.createElement("th");
			var textnode = document.createTextNode("Credits Earned");
			node.appendChild(textnode);
			document.getElementById("header").appendChild(node);
			var node = document.createElement("th");
			var textnode = document.createTextNode("Grade");
			node.appendChild(textnode);
			document.getElementById("header").appendChild(node);
			
			var head = document.getElementById("head");

			for (var j=i;j<i+8;j++){
				// var node = document.createElement("P");
				// var textnode = document.createTextNode(txt[j]);
				// node.appendChild(textnode);
				// document.getElementById("head").appendChild(node);
			}
			var count=1;
			var node = document.createElement("h5");
			var textnode = document.createTextNode(txt[i]);
			node.appendChild(textnode);
			document.getElementById("name").appendChild(node);

			var node = document.createElement("h5");
			txt[i+1] = txt[i+1].replace("_diploma","");
			var textnode = document.createTextNode(txt[i+1]);
			node.appendChild(textnode);
			document.getElementById("usn").appendChild(node);

			var node = document.createElement("h6");
			var textnode = document.createTextNode(txt[i+2]);
			node.appendChild(textnode);
			document.getElementById("degree").appendChild(node);

			var node = document.createElement("h6");
			var textnode = document.createTextNode(txt[i+3]);
			node.appendChild(textnode);
			document.getElementById("department").appendChild(node);

			// txt[i+4] = txt[i+4].replace("Credits Registered : ","");
			var node = document.createElement("b");
			var textnode = document.createTextNode(txt[i+4]);
			node.appendChild(textnode);
			document.getElementById("1").appendChild(node);

			// txt[i+5] = txt[i+5].replace("Credits Earned : ","");
			var node = document.createElement("b");
			var textnode = document.createTextNode(txt[i+5]);
			node.appendChild(textnode);
			document.getElementById("2").appendChild(node);

			txt[i+6] = txt[i+6].replace("SGPA : ","");
			var node = document.createElement("b");
			var textnode = document.createTextNode(txt[i+6]);
			node.appendChild(textnode);
			document.getElementById("rcorners1").appendChild(node);

			txt[i+7] = txt[i+7].replace("CGPA : ","");
			var node = document.createElement("b");
			var textnode = document.createTextNode(txt[i+7]);
			node.appendChild(textnode);
			document.getElementById("rcorners0").appendChild(node);


			for (var k=j;true;k=k+5)
			{
				var table = document.getElementById("mytable");
				var row = table.insertRow(count);
				var cell1 = row.insertCell(0);
				var cell2 = row.insertCell(1);
				var cell3 = row.insertCell(2);
				var cell4 = row.insertCell(3);
				var cell5 = row.insertCell(4);
				cell1.innerHTML = txt[k+1];
				cell2.innerHTML = txt[k];
				cell3.innerHTML = txt[k+2];
				cell4.innerHTML = txt[k+3];
				cell5.innerHTML = txt[k+4];

				count = count+1;

				str1 = txt[k+5].match("http");
				if (str1 == "http")
					break;

			}

			break;
		}
	}



	var flag = 1;


	function byusn(){
		loadFile2();

	}


	function loadFile2(){
		loadStrings(url,fileLoaded2);
	}

	function fileLoaded2(txt){



		string = string.replace('?usn=','');
		string = string.replace('1ms','');
		string = string.replace('1MS','');
		name = string.replace('+',' ');
		name = name.toUpperCase();
		var str1 = "1MS"
		name = str1+name;
		for(var i=0; i<txt.length;i++)
			if (name == txt[i].match(name)) {
				//console.log(txt);
				str1 =""
				flag = 0;



				var node = document.createElement("th");
				var textnode = document.createTextNode("Subject");
				node.appendChild(textnode);
				document.getElementById("header").appendChild(node);
				var node = document.createElement("th");
				var textnode = document.createTextNode("Subject Code");
				node.appendChild(textnode);
				document.getElementById("header").appendChild(node);
				var node = document.createElement("th");
				var textnode = document.createTextNode("Crdeits Required");
				node.appendChild(textnode);
				document.getElementById("header").appendChild(node);
				var node = document.createElement("th");
				var textnode = document.createTextNode("Credits Earned");
				node.appendChild(textnode);
				document.getElementById("header").appendChild(node);
				var node = document.createElement("th");
				var textnode = document.createTextNode("Grade");
				node.appendChild(textnode);
				document.getElementById("header").appendChild(node);



				for (var j=i-1;j<i+7;j++){
						// var node = document.createElement("P");
						// var textnode = document.createTextNode(txt[j]);
						// node.appendChild(textnode);
						// document.getElementById("head").appendChild(node);
					}


					var node = document.createElement("h5");
					var textnode = document.createTextNode(txt[i-1]);
					node.appendChild(textnode);
					document.getElementById("name").appendChild(node);

					var node = document.createElement("h5");
					txt[i] = txt[i].replace("_diploma","");
					var textnode = document.createTextNode(txt[i]);
					node.appendChild(textnode);
					document.getElementById("usn").appendChild(node);

					var node = document.createElement("h6");
					var textnode = document.createTextNode(txt[i+1]);
					node.appendChild(textnode);
					document.getElementById("degree").appendChild(node);

					var node = document.createElement("h6");
					var textnode = document.createTextNode(txt[i+2]);
					node.appendChild(textnode);
					document.getElementById("department").appendChild(node);

					// txt[i+3] = txt[i+3].replace("Credits Registered : ","");
					var node = document.createElement("b");
					var textnode = document.createTextNode(txt[i+3]);
					node.appendChild(textnode);
					document.getElementById("1").appendChild(node);

					// txt[i+4] = txt[i+4].replace("Credits Earned : ","");
					var node = document.createElement("b");
					var textnode = document.createTextNode(txt[i+4]);
					node.appendChild(textnode);
					document.getElementById("2").appendChild(node);

					txt[i+5] = txt[i+5].replace("SGPA : ","");
					var node = document.createElement("b");
					var textnode = document.createTextNode(txt[i+5]);
					node.appendChild(textnode);
					document.getElementById("rcorners1").appendChild(node);

					txt[i+6] = txt[i+6].replace("CGPA : ","");
					var node = document.createElement("b");
					var textnode = document.createTextNode(txt[i+6]);
					node.appendChild(textnode);
					document.getElementById("rcorners0").appendChild(node);
					var count=1;

					for (var k=j;true;k=k+5)
					{
						var table = document.getElementById("mytable");
						var row = table.insertRow(count);
						var cell1 = row.insertCell(0);
						var cell2 = row.insertCell(1);
						var cell3 = row.insertCell(2);
						var cell4 = row.insertCell(3);
						var cell5 = row.insertCell(4);
						cell1.innerHTML = txt[k+1];
						cell2.innerHTML = txt[k];
						cell3.innerHTML = txt[k+2];
						cell4.innerHTML = txt[k+3];
						cell5.innerHTML = txt[k+4];

						count = count+1;

						str1 = txt[k+5].match("http");
						if (str1 == "http")
							break;




					}








					break;
				}


				if (flag) {

					alert("No match found");
					window.open("exam.html","_self");


				};
			}






			function bybranch(){
				var x = document.getElementById("myDIV");
				if (x.style.display === "none") {
					x.style.display = "block";
				} else {
					x.style.display = "none";
				}



				loadFile3();




			}



			function loadFile3(){
				string = string.replace('?branch=','');
				string = string.replace('&sem=','');
				string = string.replace('&marks=','');
				var num = parseInt(string,10);
				n1 = int(num / 100);
				num = num - n1*100;
				n2 = int(num / 10);
				n3 = num -n2*10;

				var text = ".txt"
				var text1 = "exam_sorted/"
				// var text1 = "http://localhost:8000/web_html/exam_sorted/"
				string = string + text;
				url = text1 + string;
				loadStrings(url,fileLoaded3);
			}

			function fileLoaded3(txt){
	// console.log(txt);
	// createP(join(txt,"<br/>"));

	if(n3 == 2 || n3 == 1){

		


		var node = document.createElement("th");
		var textnode = document.createTextNode("Rank");
		node.appendChild(textnode);
		document.getElementById("header").appendChild(node);


		var node = document.createElement("th");
		var textnode = document.createTextNode("Name");
		node.appendChild(textnode);
		document.getElementById("header").appendChild(node);
		if(n3 == 1){
			var node = document.createElement("th");
			var textnode = document.createTextNode("CGPA");
			node.appendChild(textnode);
			document.getElementById("header").appendChild(node);}
			if(n3 == 2){
				var node = document.createElement("th");
				var textnode = document.createTextNode("SGPA");
				node.appendChild(textnode);
				document.getElementById("header").appendChild(node);}
				var count = 1;

				for (var k=0; k < txt.length;k=k+2)
					{	num = parseInt(txt[k+1],10);
						if(num<=10){
							var table = document.getElementById("mytable");
							var row = table.insertRow(count);
							var cell1 = row.insertCell(0);
							var cell2 = row.insertCell(1);
							var cell3 = row.insertCell(2);
							cell2.innerHTML = txt[k];
							cell3.innerHTML = txt[k+1];
							cell1.innerHTML = count;


							count = count+1;};




						}







					}
					if(n3 == 3){

						var node = document.createElement("th");
						var textnode = document.createTextNode("SL");
						node.appendChild(textnode);
						document.getElementById("header").appendChild(node);


						var node = document.createElement("th");
						var textnode = document.createTextNode("Name");
						node.appendChild(textnode);
						document.getElementById("header").appendChild(node);

						var node = document.createElement("th");
						var textnode = document.createTextNode("SGPA");
						node.appendChild(textnode);
						document.getElementById("header").appendChild(node);


						var node = document.createElement("th");
						var textnode = document.createTextNode("CGPA");
						node.appendChild(textnode);
						document.getElementById("header").appendChild(node);

						var count = 1;

						for (var k=0; k < txt.length;k=k+3)
						{
							var table = document.getElementById("mytable");
							var row = table.insertRow(count);
							var cell1 = row.insertCell(0);
							var cell2 = row.insertCell(1);
							var cell3 = row.insertCell(2);
							var cell4 = row.insertCell(3);
							cell2.innerHTML = txt[k];
							cell3.innerHTML = txt[k+1];
							cell1.innerHTML = count;
							cell4.innerHTML = txt[k+2];


							count = count+1;




						}




					}


				}


