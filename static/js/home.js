$( document ).ready(function() {

	$("#btn-start").click(function() {
		$("#btn-start").hide();
		$("#btn-stop").show();
		$("#report").hide();
		$("#webcam").webcam({
			width: 320,
			height: 240,
			mode: "callback",
			swffile: "/static/jscam_canvas_only.swf",
			onTick: function() {},
			onSave: function() {},
			onCapture: function() {},
			debug: function() {},
			onLoad: function() {}
		});
	});

	$("#btn-stop").click(function() {
		$("#btn-stop").hide();
		$("#btn-start").show();
		$("#report").show();
	});
});
