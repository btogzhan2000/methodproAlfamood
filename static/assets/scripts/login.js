    
(function ($) {
    "use strict";

    const usernameValid = "alfa@bank.com";
    const passwordValid = "alfamood2019";


    function login() {
        var email= document.getElementById("username").value;
        var password = document.getElementById("password").value;

        if(email !== usernameValid){
            if (email == ""){
                alert("Email required.");
                return ;
            }
            alert("Email does not exist.");
            return ;
        }
        else if(passwordValid != password){
            if (password == ""){
                alert("Password required.");
                return ;
            }
            alert("Password does not match.");
            return ;
        }
        else {
            document.getElementById("username").value ="";
            document.getElementById("password").value="";

            window.open("index.html");
        }

    }


    /*==================================================================
    [ Focus input ]*/
    $('.input100').each(function(){
        $(this).on('blur', function(){
            if($(this).val().trim() != "") {
                $(this).addClass('has-val');
            }
            else {
                $(this).removeClass('has-val');
            }
        })    
    })
  
  
    /*==================================================================
    [ Show pass ]*/
    var showPass = 0;
    $('.btn-show-pass').on('click', function(){
        if(showPass == 0) {
            $(this).next('input').attr('type','username');
            $(this).find('i').removeClass('zmdi-eye');
            $(this).find('i').addClass('zmdi-eye-off');
            showPass = 1;
        }
        else {
            $(this).next('input').attr('type','password');
            $(this).find('i').addClass('zmdi-eye');
            $(this).find('i').removeClass('zmdi-eye-off');
            showPass = 0;
        }
        
    });


})(jQuery);