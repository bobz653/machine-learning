#usr/bin/perl/
use Clone qw/clone/;
use Data::Dumper;

use strict ;
use warnings;


my @g_itemarr=();

my $g_item={
	'data'=>undef,'tag'=>undef
};

my @g_w=();
my $step=0.025;
my $delta=0.00001;

sub XW{
	my ($data,$w)=@_;
	my $ret=0;
	
	for(my $i=0;$i<@$data;$i++){
		$ret+=($data->[$i])*($w->[$i]);
	}
	return $ret;
}


sub Sigmod{
	my ($data,$w)=@_;
	my $val=XW($data,$w);
	return  1/(1+exp(- 1*$val));
}


sub Target{
	my ($itemarr,$w)=@_;
	my $val=0;
	for(my $i=0;$i<@$itemarr;$i++){
		 my $data=$itemarr->[$i]->{"data"};
		 my $tag=$itemarr->[$i]->{"tag"};
		 my $x=XW($data,$w);
		 $val+= $x*$tag - log(1+exp($x));
	}
	return $val;
}

sub Gradent{
	my ($itemarr,$w,$a)=@_;
	for(my $j=0;$j<@$w;$j++){
			my $tmp=0;
			for(my $i=0;$i<@$itemarr;$i++){
				 my $data=$itemarr->[$i]->{"data"};
				 my $tag=$itemarr->[$i]->{"tag"};
				 
				 $tmp+=($data->[$j])*($tag-Sigmod($data,$w));
			}
			$w->[$j]+=$a*$tmp;
	}
}

sub predict{
	my ($data,$tag,$w)=@_;
	my $val=Sigmod($data,$w);
	my $predicttag=0;
	if($val>=0.5){
		$predicttag=1;
	}
	if($predicttag == $tag){
		return 1;
	}
	return 0;
}

#原始数据
sub loaddata{
	my ($infile)=@_;
	open IN,"<$infile" or die "can't open $infile\n";
	while(<IN>){
		chomp();
		my @tmparr=split /\s+/,$_;
		my $item=clone $g_item;
		for(my $j=0;$j<$#tmparr;$j++){
			push(@{$item->{'data'}},$tmparr[$j])
		}
		$item->{'tag'}=$tmparr[$#tmparr];
		push(@g_itemarr,$item);
		
	}
	close IN;
}

#显示迭代
sub display{
	my ($cnt,$oldv,$newv,$w)=@_;
	print "iterator time: $cnt  oldv:$oldv  newv:$newv, differnt : ".abs($newv-$oldv)."\n";
	print "arguments:";
	foreach (@$w){
		print $_."\t";
	}
	print "\n";
}

loaddata("train.txt");


#N-folder
my $N=5;
my $avgprec=0;
my $partlen=int(($#g_itemarr+1)/$N);
for(my $i=0;$i<$N;$i++){
	my @traindata=();
	my @predictdata=();
	for(my $j=0;$j<=$#g_itemarr;$j++){
		if($j>=$i*$partlen && $j<=($i+1)*$partlen){
			 push(@predictdata,$g_itemarr[$j]);
		}else{
			push(@traindata,$g_itemarr[$j]); 
		}
		
	}
	print "size $#traindata  / $#predictdata \n";
	#init @w
	my $item=$g_itemarr[0]->{"data"};
	@g_w=();
	foreach(@$item){
		push(@g_w,0);
	}
	my $cnt=1;
	my $oldval=Target(\@traindata,\@g_w);
	Gradent(\@traindata,\@g_w,$step);
	my $newval=Target(\@traindata,\@g_w);
	#display($cnt,$oldval,$newval,\@g_w);
	
	while(abs($newval-$oldval)>$delta){
		$cnt++;
		$oldval=$newval;
		Gradent(\@traindata,\@g_w,$step);
		$newval=Target(\@traindata,\@g_w);
		#display($cnt,$oldval,$newval,\@g_w);
	}
	display($cnt,$oldval,$newval,\@g_w);
	print "done !\n";
	
	my $total=$#predictdata+1;
	my $rightnum=0;
	foreach(@predictdata){
		if(predict($_->{"data"},$_->{"tag"},\@g_w)==1){
			 $rightnum++;
		}
	}
	print "precsion $rightnum/$total :".$rightnum/$total."\n";
	$avgprec+=$rightnum/$total;
	
}
print "argv prec:".$avgprec/$N;





#print Dumper $g_itemarr[3];



