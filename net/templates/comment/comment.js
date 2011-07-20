<script type="text/javascript">
function addComment(form) {
    setFormStatus('comment_form', 'submitting');
    $.ajax({
        url: form.attr('action'),
        type: form.attr('method'),
        data: form.serialize(),
        dataType: 'json',
        success: function(json) {
            $('#comment_form_block').replaceWith(json.form_html);
            if (json.success) {
                if ($('#comment_list > .comment').length == 0) {
                    // hide "no comments"
                    $('#comment_list').empty();
                }
                $(json.comment_html).hide().prependTo('#comment_list').show('fast');
                
                setFormStatus('comment_form', 'success');
            }
            else {
                setFormStatus('comment_form', 'failure');
            }
        },
        error: function(xhr, ajaxOptions, thrownError) {
            setFormStatus('comment_form', 'failure');
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
