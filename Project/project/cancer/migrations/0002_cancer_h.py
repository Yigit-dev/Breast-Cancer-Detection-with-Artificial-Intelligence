# Generated by Django 2.2 on 2020-05-06 09:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cancer', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='cancer',
            name='h',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
    ]
