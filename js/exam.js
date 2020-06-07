var txt;

// function preload () {
// 	// body...
// 	txt = loadStrings("examdata.txt");

// }





function usnForm() {
	var usn = document.forms["usnform"]["usn"].value;
	if (usn == "") {
		alert("USN must be filled out");
		return false;
	}
	else{
		alert(usn);
	}
}

var name ="";
function nameForm() {
	name = document.forms["nameform"]["name"].value;
	name = name.toUpperCase();
	if (name == "") {
		alert("Name must be filled out");
		return false;
	}
	else{
		alert(name);
		
	}
}


function branchForm() {
	var branch = document.forms["branchform"]["branch"].value;
	var sem = document.forms["branchform"]["sem"].value;
	var marks = document.forms["branchform"]["marks"].value;

	if (branch == "Branch") {
		alert("Branch must be filled out");
		return false;
	}
	else{
		alert(branch);
	}


	if (sem == "Semester") {
		alert("Sem must be filled out");
		return false;
	}
	else{
		alert(sem);
	}

	if (marks == "Sorting type") {
		alert("Sorting type must be filled out");
		return false;
	}
	else{
		alert(marks);
	}
}


