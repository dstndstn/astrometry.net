<script type="text/javascript">
function setFormStatus(form_name, status) {
    form = $('form[name=' + form_name + ']')
    form.children('.form_status').children().hide();
    form.children('.form_status').children('.' + status).show();
    if (status == 'submitting') {
        form.children('input[type=submit]').attr('disabled', 'disabled');
    }
    else {
        form.children('input[type=submit]').removeAttr('disabled');
    }
}
</script>
