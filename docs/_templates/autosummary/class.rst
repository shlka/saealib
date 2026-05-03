{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :no-members:

{% block methods %}
{% if methods %}
.. rubric:: Methods

.. autosummary::
   :nosignatures:

{% for item in methods %}
   ~{{ name }}.{{ item }}
{%- endfor %}

{% endif %}
{% endblock %}

{% block method_details %}
{% if methods %}
.. rubric:: Method Details

{% for item in methods %}
.. automethod:: {{ fullname }}.{{ item }}

{% endfor %}
{% endif %}
{% endblock %}
