clear;clc;
%% 设置参数
[S1,S2]=deal(0.4,0.2);
[P,Q,R]=deal(0.1,0.3,0.5);
S3=1-S1-S2;%S3=0.2;
S=[S1,S2,S3,P,Q,R,S1*P+S2*Q+S3*R];
% 由初始参数确定的混合比例，正面概率，以及综合正面概率
%% 生成数据
N=10000;  % 数据个数
X=zeros(N,1);
PI=[P,Q,R];
for i=1:N
    n=randsrc(1,1,[1,2,3;S1,S2,S3]);  % 选择一枚硬币
    X(i)=randsrc(1,1,[1,0;PI(n),1-PI(n)]);  % 投掷这枚硬币
end
%% 设置初值
[s1,s2]=deal(0.4,0.3);
[p,q,r]=deal(0.5,0.4,0.3);
s3=1-s1-s2;
s0=[s1,s2,s3,p,q,r,s1*p+s2*q+s3*r];
% 由迭代初值确定的混合比例，正面概率，以及综合正面概率
%% EM
u1=zeros(N,1);
u2=zeros(N,1);
u3=zeros(N,1);
% p(x=1)=s1*p+s2*q+s3*r
% p(x=0)=s1*(1-p)+s2*(1-q)+s3*(1-r)
% p(x)=s1*p^x*(1-p)^(1-x)+s2*q^x*(1-q)^(1-x)+s3*r^x*(1-r)^(1-x)
M=10;  % 迭代步数
s=zeros(M,length(s0));
for j=1:M
    % E-Step
    for i=1:N
        x=X(i);
        u1(i)=(s1*p^x*(1-p)^(1-x))/(s1*p^x*(1-p)^(1-x)+s2*q^x*(1-q)^(1-x)+s3*r^x*(1-r)^(1-x));
        u2(i)=(s2*q^x*(1-q)^(1-x))/(s1*p^x*(1-p)^(1-x)+s2*q^x*(1-q)^(1-x)+s3*r^x*(1-r)^(1-x));
        u3(i)=(s3*r^x*(1-r)^(1-x))/(s1*p^x*(1-p)^(1-x)+s2*q^x*(1-q)^(1-x)+s3*r^x*(1-r)^(1-x));
    end
    % M-Step
    s1=sum(u1)/N;
    s2=sum(u2)/N;
    s3=sum(u3)/N;
    p=sum(u1.*X)/sum(u1);
    q=sum(u2.*X)/sum(u2);
    r=sum(u3.*X)/sum(u3);
    s(j,:)=[s1,s2,s3,p,q,r,s1*p+s2*q+s3*r];
    % 迭代过程中得到的混合比例，正面概率，以及综合正面概率
end 
%% Result
disp("由初始参数确定的混合比例，正面概率，以及综合正面概率"),disp(S)
disp("由迭代初值确定的混合比例，正面概率，以及综合正面概率"),disp(s0)
% disp("迭代过程中得到的混合比例，正面概率，以及综合正面概率"),disp(s)
disp("由迭代终值确定的混合比例，正面概率，以及综合正面概率"),disp(s(end,:))
disp("生成的数据中正面占比，根据迭代终值确定的正面概率")
disp([sum(X)/N,s1*p+s2*q+s3*r])
