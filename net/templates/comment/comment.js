<script type="text/javascript">
function setFormStatus(status) {
    $('#form_status').children().hide();
    $('#form_status > #' + status).show();
    if (status == 'posting') {
        $('form[name=comment_form] > input[type=submit]').attr('disabled', 'disabled');
    }
    else {
        $('form[name=comment_form] > input[type=submit]').removeAttr('disabled');
    }
}
function addComment(form) {
    setFormStatus('posting');
    $.ajax({
        url: form.attr('action'),
        type: form.attr('method'),
        data: form.serialize(),
        dataType: 'json',
        success: function(json) {
            $('#comment_form_block').replaceWith(json.form_html);
            $('#form_status').children().hide();
            if (json.success) {
                $(json.comment_html).hide().prependTo('#comment_list').show('fast');
                setFormStatus('success');
            }
            else {
                setFormStatus('failure');
            }
        },
        error: function(xhr, ajaxOptions, thrownError) {
             $('#form_status > #failure').show();
        }
    });
}
function deleteComment(comment) {
    $.ajax({
        url: comment.find('a.comment_delete').attr('href'),
        type: 'GET',
        dataType: 'json',
        success: function(json) {
            if (json.success) {
                comment.hide('fast', function() {
                    $(this).remove();
                });
            }
            else {
            }
        },
        error: function(xhr, ajaxOptions, thrownError) {
            // TODO error message
        }
    });
}

$(document).ready(function() {
    $('form[name=comment_form]').live('submit', function(event) {
        addComment($(this));
        event.preventDefault();
    });
    $('a.comment_delete').live('click', function(event) {
        deleteComment($(this).closest('div.comment'));
        event.preventDefault();
    });
});

</script>
