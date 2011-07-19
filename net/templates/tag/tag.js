<script type="text/javascript">
function addTag(form) {
    setFormStatus('tag_form', 'submitting');
    $.ajax({
        url: form.attr('action'),
        type: form.attr('method'),
        data: form.serialize(),
        dataType: 'json',
        success: function(json) {
            $('#tag_form_block').replaceWith(json.form_html);
            if (json.success) {
                $(json.tag_html).hide().appendTo('#tag_list').show('fast');
                setFormStatus('tag_form', 'success');
            }
            else {
                setFormStatus('tag_form', 'failure');
            }
        },
        error: function(xhr, ajaxOptions, thrownError) {
            setFormStatus('tag_form', 'failure');
        }
    });
}
function deleteTag(tag) {
    $.ajax({
        url: tag.find('a.tag_delete').attr('href'),
        type: 'GET',
        dataType: 'json',
        success: function(json) {
            if (json.success) {
                tag.hide('fast', function() {
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
    $('form[name=tag_form]').live('submit', function(event) {
        addTag($(this));
        event.preventDefault();
    });
    $('a.tag_delete').live('click', function(event) {
        deleteTag($(this).closest('li.tag'));
        event.preventDefault();
    });
});

</script>
