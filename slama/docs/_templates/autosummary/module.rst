.. role:: hidden
    :class: hidden-section

{{ name | underline }}

.. automodule:: {{ fullname }}

    {% block classes %}
    {% if classes %}
    .. rubric:: {{ _('Classes') }}

    .. autosummary::
        :toctree: generated
        :nosignatures:
        :template: classtemplate.rst
    {% for item in classes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    .. rubric:: {{ _('Functions') }}

    .. autosummary::
        :toctree: generated
        :nosignatures:
        :template: functiontemplate.rst
    {% for item in functions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}


{% block modules %}
{% if modules %}
.. rubric:: {{ _('Modules') }}

.. autosummary::
    :toctree:
    :recursive:
{% for item in modules %}
    {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
