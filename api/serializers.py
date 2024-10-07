from rest_framework import serializers

class AbstractSerializer(serializers.Serializer):
    abstract = serializers.CharField(required=True)
