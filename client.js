$(".submit").bind("click", function() {

	c = $(".editable.c").text();
	i = $(".editable.i").text();
	s = $(".editable.s").text();
	o = $(".editable.o").text();
	n = $(".editable.n").text();
	n_preds = $(".editable.n_preds").text();

	console.log("posting")

	$.ajax({
		url: 'http://localhost:5050/test',
		method: "POST",
		dataType: "json",
		data: JSON.stringify({"c": c, "i": i, "s": s, "o": o, "n": n, "n_preds": n_preds}),
		success: function (data){
			console.log("answer from server: ", data)
			console.log("predicted groups: " + data.groups);
			update_predictions(data.groups);
		},
		error: function( error ){
			console.log("error:", error);
		}
	})
})
	 

function update_predictions(groups){
	$(".predictions_box").empty();
	for (i in groups){
		prefix = (groups.length > 1) ? (parseInt(i)+1)+". " : "";
		$(".predictions_box").append("<p>"+prefix+groups[i]+"</p>");
	}
}
