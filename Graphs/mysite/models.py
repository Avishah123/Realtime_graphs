from django.db import models

# Create your models here.
class OptionGreeks(models.Model):
    id = models.AutoField(primary_key=True)
    strikeprice = models.FloatField()
    expirydate = models.CharField(max_length=255)
    underlying = models.CharField(max_length=255)
    identifier = models.CharField(max_length=255)
    openinterest = models.IntegerField()
    changeinopeninterest = models.IntegerField()
    pchangeinopeninterest = models.FloatField()
    totaltradedvolume = models.IntegerField()
    impliedvolatility = models.FloatField()
    lastprice = models.FloatField()
    change = models.FloatField()
    pchange = models.FloatField()
    totalbuyquantity = models.IntegerField()
    totalsellquantity = models.IntegerField()
    bidqty = models.IntegerField()
    bidprice = models.FloatField()
    askqty = models.IntegerField()
    askprice = models.FloatField()
    underlyingvalue = models.FloatField()
    instrumenttype = models.CharField(max_length=255)

    def __str__(self):
        return self.identifier  # Customize the string representation as needed

    class Meta:
        db_table = 'option_greeks'  # Optional: Specify the table name if different from the default