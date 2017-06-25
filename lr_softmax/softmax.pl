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
my @g_h=();#目标值数组
my $step=0.0025;
my $delta=0.00001;

sub EXPXW{
	my ($data,$w)=@_;
	my $ret=0;
	
	for(my $i=0;$i<@$data;$i++){
		$ret+=($data->[$i])*($w->[$i]);
	}
	return exp($ret);
}

#针对每个样例计算每个类的目标值
sub Target{
	my ($itemarr)=@_;
	#最后一个类的参数是0，所以sum是1
	my $sum=1;
	for(my $i=0;$i<@g_h;$i++){
		$g_h[$i]=EXPXW($itemarr,$g_w[$i]);
		$sum+=$g_h[$i];
	}
	for(my $i=0;$i<@g_h;$i++){
		$g_h[$i]/=$sum;
	}
}

sub Gradent{#随机梯度下降
	my ($itemarr,$w,$a)=@_;
	for(my $i=0;$i<@$itemarr;$i++){#所有训练数据，更新theta的值
		my $data=$itemarr->[$i]->{"data"};
		my $tag=$itemarr->[$i]->{"tag"};
		Target($data); #每一个样例，更新一次
		for(my $j=0;$j<@$w;$j++){#针对不同的类别
				for(my $n=0;$n<@$data;$n++){
					$w->[$j]->[$n]+=$a*$data->[$n]*(($j+1==$tag?1:0)-$g_h[$j]);
				}
		}
	}
}


#原始数据
sub loaddata{
	my ($infile)=@_;
	open IN,"<$infile" or die "can't open $infile\n";
	my %tagnumhsh=();
	my $thetanum=0;
	while(<IN>){
		chomp();
		my @tmparr=split /\,/,$_;
		my $item=clone $g_item;
		$thetanum=$#tmparr;
		for(my $j=0;$j<$#tmparr;$j++){
			push(@{$item->{'data'}},$tmparr[$j])
		}
		$item->{'tag'}=$tmparr[$#tmparr];
		$tagnumhsh{$tmparr[$#tmparr]}=1;
		push(@g_itemarr,$item);
	}
	close IN;
	for(my $i=0;$i<scalar(keys %tagnumhsh) - 1;$i++){
		my @thetaarr=();
		for(my $j=0;$j<$thetanum;$j++){
			push(@thetaarr,0.3);
		}
		push(@g_w,\@thetaarr);
		push(@g_h,0);
	}
	
	print "cat:".scalar(@g_w)."\ttheta dimension $thetanum \n";
	
}



#预测
sub predict{
	my ($data)=@_;
	Target($data->{"data"});	
	my $lastcat=0;
	foreach(@g_h){
		$lastcat+=$_;
	}
	print join(",",@g_h).",".(1-$lastcat)."\n";
}

loaddata("softmaxtrain.txt");

	print "size $#g_itemarr \n";
  my $cnt=0;
  my $opnum=100;
	while(1){
			my $break=0;
			$cnt++;
			if($cnt %100 ==0){
				print "$cnt ...\n";
			}
			if($cnt >1000){
				#last;
				}
			Gradent(\@g_itemarr,\@g_w,$step);
			
			#计算当前的目标函数值
			my $sum=0;
			foreach(@g_itemarr){
				my $data=$_->{"data"};
				my $tag=$_->{"tag"};
				my $last=1;
				for(my $i=0;$i<@g_h;$i++){
					$sum+=($i+1==$tag?1:0)*log($g_h[$i]);
					$last -=$g_h[$i];
				}
				$sum+=($#g_h+1==$tag?1:0)*log($last);
			}
			$sum/=scalar(@g_itemarr);
			$sum*=- 1;
			
			my $diff=abs($sum-$opnum);
			#print sprintf("%.12f\n",$diff);
			if($diff<$delta){
				print "iterator number : $cnt \n";
				last;
			}
			
			
			$opnum=$sum;
			
			
	}
	
	foreach(@g_itemarr){
		predict($_);
	}

#print Dumper $g_itemarr[3];



